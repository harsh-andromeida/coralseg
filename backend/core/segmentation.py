"""
SAM ViT-B prompt generation, ONNX inference execution, domain filtering, and NMS.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
from PIL import Image
import onnxruntime as ort

from core.models import _INFERENCE_LOCK


def calculate_stability_score(masks: np.ndarray, mask_threshold: float = 0.0, threshold_offset: float = 1.0) -> np.ndarray:
    """
    Computes mask stability score across +/- threshold shifts in pure NumPy.
    """
    intersections = np.sum(masks > (mask_threshold + threshold_offset), axis=(-2, -1))
    unions = np.sum(masks > (mask_threshold - threshold_offset), axis=(-2, -1))
    return np.where(unions > 0, intersections / unions, 1.0)


def nms_boxes(boxes: List[List[float]], scores: np.ndarray, iou_thresh: float = 0.7) -> List[int]:
    """Non-Maximum Suppression on bounding boxes [x0, y0, x1, y1] in pure NumPy."""
    if len(boxes) == 0:
        return []
    b = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores_arr.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def mask_nms(masks: np.ndarray, ious: np.ndarray, iou_threshold: float = 0.7) -> List[int]:
    """Non-Maximum Suppression on binary masks in pure NumPy."""
    n = len(masks)
    if n == 0:
        return []

    order = np.argsort(ious)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break

        current_mask = masks[idx]
        remaining_indices = order[1:]

        rem_masks = masks[remaining_indices]
        intersection = np.logical_and(rem_masks, current_mask).sum(axis=(-2, -1))
        union = np.logical_or(rem_masks, current_mask).sum(axis=(-2, -1))
        overlap = intersection / np.maximum(union, 1.0)

        order = remaining_indices[overlap < iou_threshold]

    return keep


def run_segmentation(
    model: Dict[str, ort.InferenceSession],
    image: Union[np.ndarray, Image.Image],
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.50,
    stability_score_thresh: float = 0.50,
    min_mask_region_area: int = 100,
    crop_n_layers: int = 0,
    box_nms_thresh: float = 0.70,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes CoralSCOP SAM dense instance segmentation using pure ONNX Runtime & NumPy.
    Implements standard ResizeLongestSide aspect-ratio preservation and prompt scaling.
    Applies CoralSCOP category classification head to reject background, fish, and open water.
    """
    from core.visualization import generate_distinct_colors

    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:
            image_np = image_np[..., :3]

    orig_h, orig_w = image_np.shape[:2]
    total_pixels = orig_h * orig_w

    sess_enc = model["encoder"]
    dec_sess = model["decoder"]

    # 1. Standard SAM ResizeLongestSide transformation (preserves aspect ratio + pads to 1024x1024)
    scale = 1024.0 / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale + 0.5), int(orig_w * scale + 0.5)

    resized_img = Image.fromarray(image_np).resize((new_w, new_h), Image.Resampling.BILINEAR)
    pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    norm_img = (np.array(resized_img, dtype=np.float32) - pixel_mean) / pixel_std

    in_enc = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    in_enc[0, :, :new_h, :new_w] = norm_img.transpose(2, 0, 1)

    with _INFERENCE_LOCK:
        image_embeddings = sess_enc.run(None, {"image": in_enc})[0]  # (1, 256, 64, 64)

    # 2. Generate regular grid of prompt points in original coordinates
    offset = 1.0 / (2 * points_per_side)
    points_1d = np.linspace(offset, 1.0 - offset, points_per_side)
    points_x, points_y = np.meshgrid(points_1d, points_1d)
    raw_grid_x = points_x.flatten() * orig_w
    raw_grid_y = points_y.flatten() * orig_h

    # Scale prompt points to the prepadded coordinate frame for SAM prompt encoder
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    scaled_pts = np.stack([raw_grid_x * scale_x, raw_grid_y * scale_y], axis=-1)
    orig_pts = np.stack([raw_grid_x, raw_grid_y], axis=-1)

    collected_masks = []
    collected_ious = []
    collected_stabs = []
    collected_boxes = []
    collected_points = []

    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0.0], dtype=np.float32)
    orig_im_size = np.array([float(orig_h), float(orig_w)], dtype=np.float32)

    # 3. Batch decode masks for grid points
    batch_size = 64
    for i in range(0, len(scaled_pts), batch_size):
        batch_pts = scaled_pts[i:i + batch_size]
        batch_orig_pts = orig_pts[i:i + batch_size]
        n_pts = len(batch_pts)

        coords = batch_pts[:, np.newaxis, :].astype(np.float32)  # (B, 1, 2)
        labels = np.ones((n_pts, 1), dtype=np.float32)          # (B, 1)

        with _INFERENCE_LOCK:
            dec_out = dec_sess.run(None, {
                "image_embeddings": image_embeddings,
                "point_coords": coords,
                "point_labels": labels,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
                "orig_im_size": orig_im_size,
            })

        batch_raw_masks = dec_out[0][:, 0]  # (B, H, W)
        batch_ious = dec_out[1][:, 0]       # (B,)

        # CoralSCOP category classification head:
        # Class 0 = Background (fish, sand, open water, macroalgae, bare rock)
        # Class 1 = True Coral instance
        if len(dec_out) >= 3 and dec_out[2].shape[-1] == 2:
            batch_cates = dec_out[2][:, 0]  # (B, 2)
            is_coral_arr = batch_cates[:, 1] > batch_cates[:, 0]
        else:
            is_coral_arr = np.ones(n_pts, dtype=bool)

        # Compute stability score from low_res_masks (output 3) if available, else full mask
        if len(dec_out) >= 4:
            batch_stabs = calculate_stability_score(dec_out[3][:, 0], mask_threshold=0.0)
        else:
            batch_stabs = calculate_stability_score(batch_raw_masks, mask_threshold=0.0)

        for j in range(n_pts):
            # Strict domain filter: Reject non-coral elements (fish, sand, water column)
            if not is_coral_arr[j]:
                continue

            iou_score = float(batch_ious[j])
            if iou_score < pred_iou_thresh:
                continue

            stab_score = float(batch_stabs[j])
            if stab_score < stability_score_thresh:
                continue

            # Binary mask at threshold 0.0
            bin_mask = batch_raw_masks[j] > 0.0
            area = int(np.sum(bin_mask))
            # Filter tiny noise fragments or giant whole-scene background masks
            if area < min_mask_region_area or area > int(0.75 * total_pixels):
                continue

            y_idx, x_idx = np.where(bin_mask)
            if len(y_idx) == 0:
                continue

            x0, x1 = float(np.min(x_idx)), float(np.max(x_idx))
            y0, y1 = float(np.min(y_idx)), float(np.max(y_idx))
            box = [x0, y0, x1, y1]

            collected_masks.append(bin_mask)
            collected_ious.append(iou_score)
            collected_stabs.append(stab_score)
            collected_boxes.append(box)
            collected_points.append(batch_orig_pts[j])

    if not collected_masks:
        return [], {
            "total_corals": 0,
            "total_corals_detected": 0,
            "coral_coverage_pct": 0.0,
            "mean_iou": 0.0,
            "mean_stability": 0.0,
            "total_pixels": total_pixels,
            "coral_pixels": 0,
        }

    masks_arr = np.stack(collected_masks, axis=0)
    ious_arr = np.array(collected_ious, dtype=np.float32)
    stabs_arr = np.array(collected_stabs, dtype=np.float32)

    # 4. Filter duplicates via Box NMS (standard SAM / CoralSCOP automatic mask generator)
    keep_indices = nms_boxes(collected_boxes, ious_arr, iou_thresh=box_nms_thresh)

    # Secondary mask IoU check if any near-identical masks remain
    if len(keep_indices) > 1:
        pruned_masks = masks_arr[keep_indices]
        pruned_ious = ious_arr[keep_indices]
        mask_keep = mask_nms(pruned_masks, pruned_ious, iou_threshold=0.85)
        keep_indices = [keep_indices[k] for k in mask_keep]

    masks_info = []
    total_coral_pixels = 0
    union_mask = np.zeros((orig_h, orig_w), dtype=bool)

    distinct_colors = generate_distinct_colors(len(keep_indices))

    for rank, idx in enumerate(keep_indices):
        m = masks_arr[idx]
        iou_val = float(ious_arr[idx])
        stab_val = float(stabs_arr[idx])
        area = int(np.sum(m))

        union_mask = np.logical_or(union_mask, m)

        # Bounding box & centroid
        y_indices, x_indices = np.where(m)
        if len(y_indices) == 0:
            continue
        x0, x1 = int(np.min(x_indices)), int(np.max(x_indices))
        y0, y1 = int(np.min(y_indices)), int(np.max(y_indices))
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))

        area_px = area
        area_pct = round((area_px / max(total_pixels, 1)) * 100.0, 2)
        color_rgb = distinct_colors[rank]
        color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
        seg_id = rank + 1

        masks_info.append({
            "id": seg_id,
            "segment_id": seg_id,
            "mask": m,
            "segmentation": m,
            "area": area_px,
            "area_px": area_px,
            "area_pct": area_pct,
            "bbox": [x0, y0, bw, bh],
            "predicted_iou": round(iou_val, 4),
            "stability_score": round(stab_val, 4),
            "centroid": (cx, cy),
            "color_rgb": color_rgb,
            "color_hex": color_hex,
        })

    coral_covered_pixels = int(np.sum(union_mask))
    coral_coverage_pct = round((coral_covered_pixels / max(total_pixels, 1)) * 100.0, 2)
    mean_iou = round(float(np.mean(ious_arr[keep_indices])), 4) if keep_indices else 0.0
    mean_stability = round(float(np.mean(stabs_arr[keep_indices])), 4) if keep_indices else 0.0

    summary_stats = {
        "total_corals": len(masks_info),
        "total_corals_detected": len(masks_info),
        "coral_coverage_pct": coral_coverage_pct,
        "mean_iou": mean_iou,
        "mean_stability": mean_stability,
        "total_pixels": total_pixels,
        "coral_pixels": coral_covered_pixels,
    }

    return masks_info, summary_stats
