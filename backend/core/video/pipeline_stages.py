"""
Self-contained, unit-testable pieces of the video pipeline used by `VideoProcessor._run_job_pipeline`
(core/video/engine.py).

  colony_color            stable colour per colony id
  labels_to_geometry      colony-id label image -> per-colony outline / bbox / centroid / area (full resolution)
  run_deliverable_exports CSV / COCO / annotated MP4 export with verification and status
  build_pipeline_timings  per-stage timing report stored on the job
"""

import os
import time
from dataclasses import dataclass, field
import colorsys
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.video.video_export import export_annotated_video, export_transect_csv, export_coco_video


def colony_color(colony_id: int) -> Tuple[int, int, int]:
    """Stable, well-separated RGB colour per colony id (golden-ratio hue walk)."""
    h = (colony_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def labels_to_geometry(labels: np.ndarray, width: int, height: int, min_pixels: int = 15) -> Dict[int, Dict[str, Any]]:
    """Per-colony geometry from a colony-id label image, in full-resolution coordinates.

    Returns {colony_id: {"bbox": [x, y, w, h], "polygon": [[x, y], ...], "centroid": (cx, cy), "area_px", "area_pct"}}.
    Colonies smaller than `min_pixels` (in label-image pixels) or without a usable outline are skipped.
    """
    ys, xs = np.nonzero(labels)
    if ys.size == 0:
        return {}
    ph, pw = labels.shape
    fx, fy = width / float(pw), height / float(ph)
    ids = labels[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids, xs, ys = ids[order], xs[order], ys[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ids)) + 1))
    counts = np.diff(np.concatenate((starts, [ids.size])))
    x0s, x1s = np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts)
    y0s, y1s = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts)
    cxs = np.add.reduceat(xs.astype(np.int64), starts) / counts
    cys = np.add.reduceat(ys.astype(np.int64), starts) / counts

    out: Dict[int, Dict[str, Any]] = {}
    for k, start in enumerate(starts):
        if counts[k] < min_pixels:
            continue
        cid = int(ids[start])
        x0, x1, y0, y1 = int(x0s[k]), int(x1s[k]), int(y0s[k]), int(y1s[k])
        crop = (labels[y0 : y1 + 1, x0 : x1 + 1] == cid).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        outline = max(contours, key=cv2.contourArea)
        if len(outline) < 3:
            continue
        polygon = [[int(round((px + x0) * fx)), int(round((py + y0) * fy))] for px, py in outline[:, 0, :]]
        area_px = int(round(int(counts[k]) * fx * fy))
        out[cid] = {
            "bbox": [int(round(x0 * fx)), int(round(y0 * fy)),
                     max(1, int(round((x1 + 1 - x0) * fx))), max(1, int(round((y1 + 1 - y0) * fy)))],
            "polygon": polygon,
            "centroid": (int(round(cxs[k] * fx)), int(round(cys[k] * fy))),
            "area_px": area_px,
            "area_pct": round(area_px / max(width * height, 1) * 100.0, 4),
        }
    return out


@dataclass
class ExportOutcome:
    paths: Dict[str, Optional[str]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    status: str = "failed"           # "ready" | "partial" | "failed"
    deliverables_sec: float = 0.0    # CSV + COCO time
    mp4_sec: float = 0.0


def _is_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def run_deliverable_exports(
    *,
    cache_dir: str,
    job_id: str,
    filename: str,
    video_path: str,
    width: int,
    height: int,
    total_frames: int,
    fps: float,
    tracks: List[Any],
    time_series_data: Any,
    frame_adapter: Any,
    export_mp4: bool,
    set_stage: Callable[..., None],
) -> ExportOutcome:
    """Writes CSV, COCO JSON and (optionally) the annotated MP4, verifying each file."""
    transect_csv = os.path.join(cache_dir, f"{job_id}_transect.csv")
    coco_json = os.path.join(cache_dir, f"{job_id}_coco_video.json")
    annotated_mp4 = os.path.join(cache_dir, f"{job_id}_annotated.mp4")

    out = ExportOutcome(paths={"annotated_mp4": None, "transect_csv": None, "coco_json": None})

    # 1. CSV
    t_exp_d = time.perf_counter()
    try:
        export_transect_csv(output_csv_path=transect_csv, time_series_records=time_series_data)
        if _is_nonempty(transect_csv):
            out.paths["transect_csv"] = transect_csv
        else:
            out.errors["transect_csv"] = "CSV file was not created or has 0 bytes"
    except Exception as csv_err:
        out.errors["transect_csv"] = str(csv_err)
        set_stage(log_entry=f"CSV export error: {csv_err}")

    # 2. COCO JSON
    try:
        export_coco_video(
            output_json_path=coco_json,
            video_filename=filename,
            width=width,
            height=height,
            total_frames=total_frames,
            fps=fps,
            all_tracks=tracks,
            frame_tracks_map=frame_adapter,
        )
        if _is_nonempty(coco_json):
            out.paths["coco_json"] = coco_json
        else:
            out.errors["coco_json"] = "COCO JSON file was not created or has 0 bytes"
    except Exception as coco_err:
        out.errors["coco_json"] = str(coco_err)
        set_stage(log_entry=f"COCO JSON export error: {coco_err}")
    out.deliverables_sec = time.perf_counter() - t_exp_d

    # 3. Optional MP4 overlay
    if export_mp4:
        t_exp_m = time.perf_counter()
        try:
            set_stage(stage="Rendering annotated video overlay...", action="Encoding H.264 preview video")
            export_annotated_video(
                input_video_path=video_path,
                output_mp4_path=annotated_mp4,
                frame_tracks_map=frame_adapter,
            )
            if _is_nonempty(annotated_mp4):
                out.paths["annotated_mp4"] = annotated_mp4
            else:
                out.errors["annotated_mp4"] = "Annotated MP4 file was not created or has 0 bytes"
        except Exception as mp4_err:
            out.errors["annotated_mp4"] = str(mp4_err)
            set_stage(log_entry=f"MP4 export error: {mp4_err}")
        out.mp4_sec = time.perf_counter() - t_exp_m

    expected = ["transect_csv", "coco_json"] + (["annotated_mp4"] if export_mp4 else [])
    ok = [k for k in expected if out.paths.get(k) is not None]
    out.status = "ready" if len(ok) == len(expected) else ("partial" if ok else "failed")
    return out


def build_pipeline_timings(
    *,
    elapsed_total: float,
    model_load_sec: float,
    decode_sec: float,
    ego_motion_sec: float,
    flow_sec: float,
    geometry_sec: float,
    keyframe_blocked_sec: float,
    keyframe_segmentation_sec: float,
    classification_sec: float,
    persistence_sec: float,
    export_deliverables_sec: float,
    export_mp4_sec: float,
    unique_colonies: int,
    keyframes_count: int,
    committed_frames: int,
) -> Dict[str, Any]:
    """Builds the per-stage timing report (seconds and percent of elapsed) stored on the job.

    Keyframe segmentation and classification run in background threads, so their seconds overlap the main loop
    and are not part of the critical path; `keyframe_blocked_sec` is the time the main loop actually waited for one.
    """
    critical_path = (
        decode_sec + ego_motion_sec + flow_sec + geometry_sec + keyframe_blocked_sec
        + persistence_sec + export_deliverables_sec + export_mp4_sec
    )
    idle = max(0.0, elapsed_total - critical_path)
    denom = max(elapsed_total, 0.001)

    def pct(v: float) -> float:
        return round((v / denom) * 100.0, 1)

    return {
        "total_elapsed_sec": round(elapsed_total, 3),
        "model_load_sec": round(model_load_sec, 3),
        "video_decode_sec": round(decode_sec, 3),
        "ego_motion_sec": round(ego_motion_sec, 3),
        "optical_flow_and_warp_sec": round(flow_sec, 3),
        "geometry_extraction_sec": round(geometry_sec, 3),
        "blocked_on_keyframe_sec": round(keyframe_blocked_sec, 3),
        "keyframe_segmentation_sec": round(keyframe_segmentation_sec, 3),
        "classification_sec": round(classification_sec, 3),
        "persistence_sec": round(persistence_sec, 3),
        "export_deliverables_sec": round(export_deliverables_sec, 3),
        "export_annotated_mp4_sec": round(export_mp4_sec, 3),
        "unaccounted_idle_sec": round(idle, 3),
        "pct_video_decode": pct(decode_sec),
        "pct_optical_flow_and_warp": pct(flow_sec),
        "pct_geometry": pct(geometry_sec),
        "pct_persistence": pct(persistence_sec),
        "pct_export": pct(export_deliverables_sec + export_mp4_sec),
        "pct_unaccounted_idle": pct(idle),
        "unique_colonies_count": unique_colonies,
        "keyframes_count": keyframes_count,
        "committed_frames": committed_frames,
    }
