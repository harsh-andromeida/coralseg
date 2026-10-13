"""
Visualization functions for high-contrast segmentation masks, contours, bounding boxes, and labels.
Pure NumPy and Pillow implementation with zero OpenCV dependencies.
"""

import colorsys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generates n visually distinct, vibrant RGB colors using golden-ratio hue distribution.
    """
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    hue = 0.12

    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1.0
        saturation = 0.85
        value = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


@lru_cache(maxsize=32)
def get_overlay_font(size: int) -> ImageFont.ImageFont:
    """Returns a cached TrueType font scaled to the requested size with cross-platform fallbacks."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "Arial Bold",
        "Arial",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "helvetica",
    ]
    for cand in candidates:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def get_mask_overlay_color(mask_dict: Dict[str, Any], color_mode: str = "instance") -> Tuple[int, int, int]:
    """Resolves the proper RGB color for a mask based on the selected overlay color mode."""
    if color_mode in ("taxonomy", "taxonomy_condition"):
        tax = mask_dict.get("taxonomy", {})
        if "color" in tax:
            return tuple(tax["color"])
        if "taxonomy_color_rgb" in mask_dict:
            return tuple(mask_dict["taxonomy_color_rgb"])
        return mask_dict.get("color_rgb", (0, 204, 255))
    elif color_mode == "bleaching":
        if mask_dict.get("bleaching", {}).get("is_bleached"):
            return (239, 68, 68)  # Coral red for bleached
        return (16, 185, 129)      # Emerald green for healthy
    else:  # "instance"
        return mask_dict.get("instance_color_rgb", mask_dict.get("color_rgb", mask_dict.get("color", (0, 255, 200))))


def get_mask_overlay_label(mask_dict: Dict[str, Any], color_mode: str = "instance", is_selected: bool = False) -> str:
    """Generates the badge text for a mask based on the selected overlay color mode."""
    mask_id = mask_dict.get("id", mask_dict.get("segment_id", 1))
    tax = mask_dict.get("taxonomy", {})
    genus = tax.get("genus", "Coral")
    bleach = mask_dict.get("bleaching", {})
    is_bleached = bleach.get("is_bleached", False)
    cond_str = "Bleached" if is_bleached else "Healthy"

    if color_mode == "taxonomy":
        if is_selected:
            return f"#{mask_id} {genus}"
        return genus
    elif color_mode == "taxonomy_condition":
        return f"#{mask_id} {genus} • {cond_str}"
    elif color_mode == "bleaching":
        return f"#{mask_id} {cond_str}"
    else:  # "instance"
        return f"#{mask_id}"


def create_segmentation_overlay(
    image: Union[np.ndarray, Image.Image],
    masks_info: List[Dict[str, Any]],
    alpha: float = 0.45,
    draw_contours: bool = True,
    draw_labels: bool = True,
    draw_boxes: bool = False,
    selected_mask_id: Optional[int] = None,
    color_mode: str = "instance",
    draw_centroids: bool = False,
    draw_ids: Optional[bool] = None,
    contour_thickness: int = 2,
    custom_color_mode: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Renders high-contrast, translucent color overlays on top of the original image.
    Pure NumPy & PIL rendering with zero OpenCV dependencies.
    """
    if draw_ids is not None:
        draw_labels = draw_ids
    if custom_color_mode is not None:
        color_mode = custom_color_mode

    if isinstance(image, Image.Image):
        base_np = np.array(image.convert("RGB"))
    else:
        base_np = np.array(image).copy()
        if len(base_np.shape) == 2:
            base_np = np.stack([base_np] * 3, axis=-1)
        elif base_np.shape[2] == 4:
            base_np = base_np[..., :3]

    h, w = base_np.shape[:2]
    scale = max(h, w) / 1000.0
    blended = base_np.copy()

    # Paint translucent overlay segments
    for mask_dict in masks_info:
        seg = mask_dict.get("mask", mask_dict.get("segmentation"))
        if seg is None:
            continue
        if isinstance(seg, np.ndarray) and seg.dtype != bool:
            seg = seg.astype(bool)

        mask_id = mask_dict.get("id", mask_dict.get("segment_id", 1))
        color = get_mask_overlay_color(mask_dict, color_mode)

        curr_alpha = alpha
        if selected_mask_id is not None:
            if mask_id == selected_mask_id:
                curr_alpha = min(alpha + 0.25, 0.85)
            else:
                curr_alpha = alpha * 0.20

        colored = np.array(color, dtype=np.float32)
        orig_part = blended[seg].astype(np.float32)
        blended[seg] = (colored * curr_alpha + orig_part * (1.0 - curr_alpha)).clip(0, 255).astype(np.uint8)

    # Draw high-contrast crisp segment borders using 4-connected boundary
    if draw_contours:
        for mask_dict in masks_info:
            seg = mask_dict.get("mask", mask_dict.get("segmentation"))
            if seg is None:
                continue
            if isinstance(seg, np.ndarray) and seg.dtype != bool:
                seg = seg.astype(bool)

            mask_id = mask_dict.get("id", mask_dict.get("segment_id", 1))
            color = get_mask_overlay_color(mask_dict, color_mode)

            padded = np.pad(seg, 1, mode='constant', constant_values=False)
            eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
            edge = seg & ~eroded

            is_selected = (selected_mask_id is not None and mask_id == selected_mask_id)
            border_color = (255, 255, 255) if is_selected else color

            if contour_thickness > 1 or is_selected:
                p_edge = np.pad(edge, 1, mode='constant', constant_values=False)
                dilated_edge = (p_edge[1:-1, 1:-1] | p_edge[:-2, 1:-1] | p_edge[2:, 1:-1] | p_edge[1:-1, :-2] | p_edge[1:-1, 2:])
                blended[dilated_edge] = border_color
            else:
                blended[edge] = border_color

    out_pil = Image.fromarray(blended)
    if draw_boxes or draw_centroids or draw_labels:
        draw = ImageDraw.Draw(out_pil)

        if draw_boxes:
            box_width = max(1, int(round(1.5 * scale)))
            selected_bw = max(2, int(round(3.0 * scale)))
            for mask_dict in masks_info:
                bbox = mask_dict.get("bbox")
                if not bbox:
                    continue
                x, y, bw, bh = [int(v) for v in bbox]
                mask_id = mask_dict.get("id", mask_dict.get("segment_id", 1))
                color = get_mask_overlay_color(mask_dict, color_mode)
                is_selected = (selected_mask_id is not None and mask_id == selected_mask_id)
                box_color = (255, 255, 255) if is_selected else color
                curr_bw = selected_bw if is_selected else box_width
                draw.rectangle([x, y, x + bw, y + bh], outline=box_color, width=curr_bw)

        if draw_centroids:
            r = max(2, int(round(3.5 * scale)))
            cw = max(1, int(round(scale)))
            for mask_dict in masks_info:
                cx, cy = mask_dict.get("centroid", (0, 0))
                if 0 <= cx < w and 0 <= cy < h:
                    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 255, 255), outline=(0, 0, 0), width=cw)

        if draw_labels:
            font_size = max(11, int(round(12 * scale)))
            font = get_overlay_font(font_size)
            selected_font = get_overlay_font(max(13, int(round(15 * scale))))

            pad_x = max(3, int(round(5 * scale)))
            pad_y = max(2, int(round(3 * scale)))
            radius = max(3, int(round(4 * scale)))
            border_w = max(1, int(round(1.2 * scale)))

            for mask_dict in masks_info:
                mask_id = mask_dict.get("id", mask_dict.get("segment_id", 1))
                if selected_mask_id is not None and selected_mask_id != mask_id:
                    continue
                cx, cy = mask_dict.get("centroid", (0, 0))
                if 0 <= cx < w and 0 <= cy < h:
                    is_selected = (selected_mask_id is not None and mask_id == selected_mask_id)
                    curr_font = selected_font if is_selected else font

                    color = get_mask_overlay_color(mask_dict, color_mode)
                    txt = get_mask_overlay_label(mask_dict, color_mode, is_selected=is_selected)

                    bbox = draw.textbbox((cx, cy), txt, font=curr_font, anchor="mm")
                    pill_box = [
                        bbox[0] - pad_x,
                        bbox[1] - pad_y,
                        bbox[2] + pad_x,
                        bbox[3] + pad_y,
                    ]

                    if is_selected:
                        draw.rounded_rectangle(
                            pill_box,
                            radius=radius,
                            fill=(13, 124, 133),
                            outline=(255, 255, 255),
                            width=max(2, int(round(2 * scale))),
                        )
                        draw.text((cx, cy), txt, font=curr_font, fill=(255, 255, 255), anchor="mm")
                    else:
                        draw.rounded_rectangle(
                            pill_box,
                            radius=radius,
                            fill=(15, 23, 42),
                            outline=color,
                            width=border_w,
                        )
                        draw.text((cx, cy), txt, font=curr_font, fill=(255, 255, 255), anchor="mm")

    return np.array(out_pil)
