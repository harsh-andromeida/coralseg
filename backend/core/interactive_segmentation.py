"""
Interactive SAM Prompt Segmentation, Moore-Neighbor Contour Extraction,
RDP Polygon Simplification, and PIL Polygon Rasterization.
Zero OpenCV dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

from core.models import _INFERENCE_LOCK, load_coralscop_model


def ramer_douglas_peucker(points: List[Tuple[int, int]], epsilon: float) -> List[Tuple[int, int]]:
    """
    Ramer-Douglas-Peucker (RDP) algorithm for polygonal curve simplification.
    Reduces redundant dense points into clean, draggable polygon handles.
    """
    if len(points) < 3:
        return points

    pts = np.array(points, dtype=np.float32)
    start = pts[0]
    end = pts[-1]

    # Vector from start to end
    line_vec = end - start
    line_len_sq = np.sum(line_vec ** 2)

    if line_len_sq == 0:
        dists = np.sqrt(np.sum((pts - start) ** 2, axis=1))
    else:
        # Perpendicular distance from each point to the line segment
        t = np.clip(np.sum((pts - start) * line_vec, axis=1) / line_len_sq, 0.0, 1.0)
        proj = start + np.outer(t, line_vec)
        dists = np.sqrt(np.sum((pts - proj) ** 2, axis=1))

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left_sub = ramer_douglas_peucker(points[: max_idx + 1], epsilon)
        right_sub = ramer_douglas_peucker(points[max_idx:], epsilon)
        return left_sub[:-1] + right_sub
    else:
        return [points[0], points[-1]]


def mask_to_polygon(
    mask: np.ndarray,
    tolerance: float = 2.0,
    min_points: int = 4,
    max_points: int = 100,
) -> List[List[int]]:
    """
    Extracts the outer boundary polygon of a binary mask using Moore-Neighbor Tracing,
    and simplifies the vertex count using Ramer-Douglas-Peucker (RDP).
    Returns list of [x, y] coordinates.
    """
    if mask is None or not np.any(mask):
        return []

    h, w = mask.shape[:2]
    # Find start point (topmost, then leftmost)
    y_idx, x_idx = np.where(mask)
    if len(y_idx) == 0:
        return []

    start_y, start_x = int(y_idx[0]), int(x_idx[0])

    # 8-connected clockwise directions: N, NE, E, SE, S, SW, W, NW
    dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    raw_contour: List[Tuple[int, int]] = [(start_x, start_y)]

    curr_p = (start_y, start_x)
    backtrack_dir = 6  # Entered from West (direction index 6)
    max_steps = max(1000, 4 * (h + w))

    for _ in range(max_steps):
        found = False
        for step in range(8):
            d_idx = (backtrack_dir + step) % 8
            dy, dx = dirs[d_idx]
            ny, nx = curr_p[0] + dy, curr_p[1] + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                curr_p = (ny, nx)
                raw_contour.append((nx, ny))
                backtrack_dir = (d_idx + 4 + 1) % 8
                found = True
                break
        if not found or curr_p == (start_y, start_x):
            break

    if len(raw_contour) < min_points:
        return [[int(x), int(y)] for x, y in raw_contour]

    # Simplify with RDP
    simplified = ramer_douglas_peucker(raw_contour, epsilon=tolerance)

    # If still too dense for high-fps canvas editing, subsample evenly
    if len(simplified) > max_points:
        step = max(1, len(simplified) // max_points)
        simplified = simplified[::step]

    # Ensure closed polygon loop
    if len(simplified) >= 3 and (simplified[0] != simplified[-1]):
        simplified.append(simplified[0])

    return [[int(pt[0]), int(pt[1])] for pt in simplified]


def polygon_to_mask(polygon: List[List[Union[int, float]]], height: int, width: int) -> np.ndarray:
    """
    Rasterizes a list of [x, y] polygon vertices into a boolean mask of shape (height, width).
    Pure PIL implementation.
    """
    if not polygon or len(polygon) < 3:
        return np.zeros((height, width), dtype=bool)

    pts = [(float(p[0]), float(p[1])) for p in polygon]
    img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def calculate_polygon_metrics(
    polygon: List[List[Union[int, float]]],
    height: int,
    width: int,
) -> Dict[str, Any]:
    """
    Computes exact pixel area, area %, bounding box [x, y, w, h], and centroid (cx, cy).
    """
    mask = polygon_to_mask(polygon, height, width)
    total_pixels = max(height * width, 1)
    area_px = int(np.sum(mask))
    area_pct = round((area_px / total_pixels) * 100.0, 2)

    if area_px == 0:
        # Fallback to polygon vertices extent
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        min_x, max_x = min(xs, default=0), max(xs, default=0)
        min_y, max_y = min(ys, default=0), max(ys, default=0)
        bbox = [int(min_x), int(min_y), max(1, int(max_x - min_x)), max(1, int(max_y - min_y))]
        centroid = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
    else:
        y_idx, x_idx = np.where(mask)
        x0, x1 = int(np.min(x_idx)), int(np.max(x_idx))
        y0, y1 = int(np.min(y_idx)), int(np.max(y_idx))
        bbox = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
        centroid = (int(np.mean(x_idx)), int(np.mean(y_idx)))

    return {
        "mask": mask,
        "area_px": area_px,
        "area_pct": area_pct,
        "bbox": bbox,
        "centroid": centroid,
    }


class InteractiveSAMSession:
    """
    Manages cached SAM image embeddings and point prompt inferences.
    Computes encoder once per image; decoder runs in ~15-25ms per click.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def has_embeddings(self, image_name: str) -> bool:
        """Returns True if embeddings for image_name are already cached."""
        return image_name in self._cache

    def set_embeddings(self, image_name: str, entry: Dict[str, Any]) -> None:
        """Stores precomputed SAM embeddings for image_name."""
        self._cache[image_name] = entry

    def get_or_compute_embeddings(
        self,
        sam_bundle: Dict[str, ort.InferenceSession],
        image_name: str,
        pil_image: Image.Image,
    ) -> Dict[str, Any]:
        """Retrieves or executes SAM ViT-B image encoder for image_name."""
        if image_name in self._cache:
            return self._cache[image_name]

        # Double-checked locking to avoid duplicate encoder runs during concurrent requests
        with _INFERENCE_LOCK:
            if image_name in self._cache:
                return self._cache[image_name]

            orig_w, orig_h = pil_image.width, pil_image.height
            scale = 1024.0 / max(orig_h, orig_w)
            new_h, new_w = int(orig_h * scale + 0.5), int(orig_w * scale + 0.5)

            # Direct PIL resize without redundant numpy-to-PIL conversion
            pil_rgb = pil_image.convert("RGB")
            resized_img = pil_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
            pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            norm_img = (np.asarray(resized_img, dtype=np.float32) - pixel_mean) / pixel_std

            in_enc = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
            in_enc[0, :, :new_h, :new_w] = norm_img.transpose(2, 0, 1)

            sess_enc = sam_bundle["encoder"]
            image_embeddings = sess_enc.run(None, {"image": in_enc})[0]

            entry = {
                "image_embeddings": image_embeddings,
                "orig_h": orig_h,
                "orig_w": orig_w,
                "new_h": new_h,
                "new_w": new_w,
                "scale": scale,
            }
            self._cache[image_name] = entry
            return entry

    def prompt_segment(
        self,
        sam_bundle: Dict[str, ort.InferenceSession],
        image_name: str,
        pil_image: Image.Image,
        points: List[Dict[str, Any]],  # [{"x": float, "y": float, "label": int}]
    ) -> Dict[str, Any]:
        """
        Executes interactive point prompt segmentation.
        Points have label 1 (positive foreground) or 0 (negative background).
        Supports multi-point prompting via iterative mask refinement conforming to
        the ONNX decoder input shape constraints: point_coords (1, 1, 2) and point_labels (1, 1).
        """
        if not points:
            raise ValueError("No prompt points provided")

        pos_points = [p for p in points if p.get("label", 1) == 1]
        if not pos_points:
            raise ValueError("At least one foreground (+) prompt pin is required to segment an object.")

        cached = self.get_or_compute_embeddings(sam_bundle, image_name, pil_image)
        orig_h, orig_w = cached["orig_h"], cached["orig_w"]
        new_h, new_w = cached["new_h"], cached["new_w"]
        image_embeddings = cached["image_embeddings"]

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        # Sort points: positive foreground pins (label 1) first to establish the initial
        # instance mask, followed by negative background pins (label 0) for boundary refinement.
        sorted_points = sorted(points, key=lambda p: (0 if p.get("label", 1) == 1 else 1))

        low_res_mask = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)
        orig_im_size = np.array([float(orig_h), float(orig_w)], dtype=np.float32)

        dec_sess = sam_bundle["decoder"]
        final_bin_mask = None
        final_iou = 0.0

        with _INFERENCE_LOCK:
            for pt in sorted_points:
                px = float(pt["x"]) * scale_x
                py = float(pt["y"]) * scale_y
                coords_arr = np.array([[[px, py]]], dtype=np.float32)  # (1, 1, 2)
                labels_arr = np.array([[float(pt.get("label", 1))]], dtype=np.float32)  # (1, 1)

                dec_out = dec_sess.run(None, {
                    "image_embeddings": image_embeddings,
                    "point_coords": coords_arr,
                    "point_labels": labels_arr,
                    "mask_input": low_res_mask,
                    "has_mask_input": has_mask_input,
                    "orig_im_size": orig_im_size,
                })

                raw_masks = dec_out[0][0]  # (K, orig_h, orig_w) or (orig_h, orig_w)
                iou_preds = dec_out[1][0]  # (K,) or scalar
                if raw_masks.ndim == 3:
                    best_k = int(np.argmax(iou_preds))
                    raw_mask = raw_masks[best_k]
                    final_iou = float(iou_preds[best_k])
                    low_res_mask = dec_out[3][:, best_k:best_k+1]
                else:
                    raw_mask = raw_masks
                    final_iou = float(iou_preds[0]) if hasattr(iou_preds, "__getitem__") else float(iou_preds)
                    low_res_mask = dec_out[3]

                has_mask_input = np.array([1.0], dtype=np.float32)
                final_bin_mask = raw_mask > 0.0

        # Convert binary mask to clean polygon vertices
        polygon = mask_to_polygon(final_bin_mask, tolerance=1.8, max_points=100)
        metrics = calculate_polygon_metrics(polygon, orig_h, orig_w)

        return {
            "polygon": polygon,
            "area_px": metrics["area_px"],
            "area_pct": metrics["area_pct"],
            "bbox": metrics["bbox"],
            "centroid": metrics["centroid"],
            "predicted_iou": round(final_iou, 4),
        }

    def clear(self):
        """Clears cached embeddings."""
        self._cache.clear()


# Global singleton instance
SAM_SESSION = InteractiveSAMSession()
