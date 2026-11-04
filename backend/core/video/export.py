"""
Video Overlay Rendering, MP4 Compilation, Transect CSV Generation, and MOT/COCO-Video Serialization.
Utilizes fast vectorized OpenCV drawing operations for real-time video playback and export.
"""

import os
import json
import csv
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from core.video.tracker import CoralTrack


def render_video_overlay(
    frame_bgr: np.ndarray,
    active_tracks: List[CoralTrack],
    draw_polygons: bool = True,
    draw_boxes: bool = False,
    draw_labels: bool = True,
    draw_hud: bool = True,
    frame_idx: int = 0,
    total_frames: int = 0,
    fps: float = 30.0,
    bleaching_prevalence_pct: float = 0.0,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Renders high-contrast segmentation masks, bounding boxes, species tags, and benthic HUD
    directly onto an OpenCV BGR video frame in ~2-4ms.
    """
    if not _HAS_CV2 or len(active_tracks) == 0 and not draw_hud:
        return frame_bgr

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # 1. Render Polygons with Alpha Blending
    if draw_polygons:
        overlay = out.copy()
        for track in active_tracks:
            if not track.polygon or len(track.polygon) < 3:
                continue

            pts = np.array(track.polygon, dtype=np.int32).reshape((-1, 1, 2))
            # Track color is RGB; convert to BGR for OpenCV
            color_bgr = (int(track.color_rgb[2]), int(track.color_rgb[1]), int(track.color_rgb[0]))

            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.polylines(out, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    # 2. Render Bounding Boxes & Labels
    for track in active_tracks:
        x, y, bw, bh = track.bbox
        color_bgr = (int(track.color_rgb[2]), int(track.color_rgb[1]), int(track.color_rgb[0]))

        if draw_boxes:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color_bgr, thickness=2, lineType=cv2.LINE_AA)

        if draw_labels:
            genus = track.taxonomy.get("genus", "Coral")
            is_bleached = track.bleaching.get("is_bleached", False)
            cond_str = "Bleached" if is_bleached else "Healthy"
            label = f"#{track.track_id} {genus} [{cond_str}]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_x = max(0, x)
            badge_y = max(text_h + 4, y - 6)

            # Draw background pill
            badge_bg = (20, 20, 20)
            cv2.rectangle(
                out,
                (badge_x, badge_y - text_h - 4),
                (badge_x + text_w + 6, badge_y + 2),
                badge_bg,
                -1,
            )
            # Text colored red if bleached, else track color
            text_color = (68, 68, 239) if is_bleached else (240, 240, 240)
            cv2.putText(
                out,
                label,
                (badge_x + 3, badge_y - 2),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA,
            )

    # 3. Render Top Benthic HUD
    if draw_hud:
        hud_h = 36
        hud_overlay = out[:hud_h, :].copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (12, 16, 24), -1)
        cv2.addWeighted(hud_overlay, 0.85, out[:hud_h, :], 0.15, 0, out[:hud_h, :])

        sec = frame_idx / max(fps, 1.0)
        hud_text = (
            f"ReefVision Transect Monitor | Frame {frame_idx + 1}/{total_frames} ({sec:.1f}s) | "
            f"Active Corals: {len(active_tracks)} | "
            f"Bleaching: {bleaching_prevalence_pct:.1f}% | "
            f"{fps:.1f} FPS"
        )
        cv2.putText(
            out,
            hud_text,
            (14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 240, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return out


def export_annotated_video(
    input_video_path: str,
    output_mp4_path: str,
    frame_tracks_map: Dict[int, List[CoralTrack]],
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    max_frames: Optional[int] = None,
) -> str:
    """
    Generates an annotated MP4 video with rendered bounding boxes, polygons, and benthic HUD.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required to export video.")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {input_video_path}")

    raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(raw_total_frames, max_frames) if max_frames is not None and max_frames > 0 else raw_total_frames
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

    try:
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            tracks = frame_tracks_map.get(frame_idx, [])
            bleached_count = sum(1 for t in tracks if t.bleaching.get("is_bleached", False))
            bleach_pct = (bleached_count / max(len(tracks), 1)) * 100.0

            rendered = render_video_overlay(
                frame_bgr=frame_bgr,
                active_tracks=tracks,
                draw_polygons=True,
                draw_boxes=False,
                draw_labels=True,
                draw_hud=True,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
                bleaching_prevalence_pct=bleach_pct,
            )
            writer.write(rendered)

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx, total_frames, (frame_idx / total_frames) * 100.0)

        return output_mp4_path
    finally:
        cap.release()
        writer.release()


def export_transect_csv(
    output_csv_path: str,
    time_series_records: List[Dict[str, Any]],
) -> str:
    """
    Exports frame-by-frame scientific transect statistics to CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "active_corals_count",
        "cumulative_unique_corals",
        "bleached_corals_count",
        "healthy_corals_count",
        "bleaching_prevalence_pct",
        "dominant_genus",
        "is_keyframe",
        "camera_motion_dx",
        "camera_motion_dy",
    ]

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in time_series_records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})

    return output_csv_path


def export_coco_video(
    output_json_path: str,
    video_filename: str,
    width: int,
    height: int,
    total_frames: int,
    fps: float,
    all_tracks: List[CoralTrack],
    frame_tracks_map: Dict[int, List[CoralTrack]],
) -> str:
    """
    Exports tracking annotations to MOT / COCO-Video format JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)

    videos = [{
        "id": 1,
        "name": video_filename,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
    }]

    categories = [
        {"id": 1, "name": "Coral Instance", "supercategory": "Benthos"},
    ]

    annotations = []
    ann_id = 1
    for frame_idx, tracks in frame_tracks_map.items():
        for t in tracks:
            annotations.append({
                "id": ann_id,
                "video_id": 1,
                "frame_id": frame_idx + 1,
                "track_id": t.track_id,
                "category_id": 1,
                "bbox": t.bbox,
                "area": t.area_px,
                "predicted_iou": t.predicted_iou,
                "taxonomy": t.taxonomy.get("genus", "Coral"),
                "growth_form": t.taxonomy.get("growth_form", "-"),
                "condition": t.bleaching.get("status", "Healthy"),
                "is_bleached": t.bleaching.get("is_bleached", False),
            })
            ann_id += 1

    payload = {
        "info": {
            "description": "Andromeida ReefVision Video Transect Tracking",
            "version": "2.0",
        },
        "videos": videos,
        "categories": categories,
        "annotations": annotations,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_json_path
