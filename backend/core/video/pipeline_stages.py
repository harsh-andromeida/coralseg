"""
Self-contained, unit-testable pieces of the video pipeline, extracted from
`VideoProcessor._run_job_pipeline` (core/video/engine.py) without behaviour change.

  rescale_detections_to_full_res  proxy-resolution detections -> full-resolution coordinates
  track_metadata                  metadata dict handed to the SAM tracker when registering a mask
  crop_track_regions              padded crops of tracked colonies for BioCLIP / bleaching
  run_deliverable_exports         CSV / COCO / annotated MP4 export with verification and status
  build_pipeline_timings          per-stage timing report stored on the job
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from core.interactive_segmentation import mask_to_polygon
from core.video.video_export import export_annotated_video, export_transect_csv, export_coco_video


def rescale_detections_to_full_res(masks_info: List[Dict[str, Any]], scale_x: float, scale_y: float) -> None:
    """Remaps proxy-space detections (polygon, bbox, centroid, areas) to full resolution, in place."""
    for m in masks_info:
        # run_segmentation yields masks without polygons; extract them in proxy
        # space here so they get remapped like bbox/centroid below. Otherwise
        # the tracker derives them later from the proxy mask, unscaled.
        if not m.get("polygon") and m.get("mask") is not None:
            m["polygon"] = mask_to_polygon(m["mask"], tolerance=1.5)
        if "polygon" in m and m["polygon"]:
            m["polygon"] = [
                [int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))]
                for pt in m["polygon"]
            ]
        if "bbox" in m:
            bx, by, bw, bh = m["bbox"]
            m["bbox"] = [
                int(round(bx * scale_x)),
                int(round(by * scale_y)),
                max(1, int(round(bw * scale_x))),
                max(1, int(round(bh * scale_y))),
            ]
        if "centroid" in m:
            cx, cy = m["centroid"]
            m["centroid"] = (int(round(cx * scale_x)), int(round(cy * scale_y)))
        if "area" in m:
            m["area"] = int(round(m["area"] * scale_x * scale_y))
        if "area_px" in m:
            m["area_px"] = int(round(m["area_px"] * scale_x * scale_y))


def track_metadata(trk: Any, first_seen_frame: Optional[int] = None) -> Dict[str, Any]:
    """Metadata registered with the SAM tracker alongside a track's mask."""
    return {
        "track_id": trk.track_id,
        "genus": trk.taxonomy.get("genus", "Coral"),
        "growth_form": trk.taxonomy.get("growth_form", "-"),
        "taxon_conf": trk.taxonomy.get("confidence_pct", 85.0),
        "condition": trk.bleaching.get("status", "Healthy"),
        "condition_conf": trk.bleaching.get("confidence_pct", 90.0),
        "is_bleached": trk.bleaching.get("is_bleached", False),
        "color_hex": trk.color_hex,
        "first_seen_frame": trk.first_seen_frame if first_seen_frame is None else first_seen_frame,
        "predicted_iou": trk.predicted_iou,
    }


def crop_track_regions(pil_img: Image.Image, tracks: List[Any], width: int, height: int) -> List[Image.Image]:
    """One crop per track (bbox padded by 10%, minimum 10px), in the same order as `tracks`."""
    crops = []
    for trk in tracks:
        x, y, bw, bh = trk.bbox
        x0 = max(0, x - int(bw * 0.1))
        y0 = max(0, y - int(bh * 0.1))
        x1 = min(width, x + bw + int(bw * 0.1))
        y1 = min(height, y + bh + int(bh * 0.1))
        if (x1 - x0) >= 10 and (y1 - y0) >= 10:
            crops.append(pil_img.crop((x0, y0, x1, y1)))
        else:
            crops.append(pil_img.crop((x, y, max(x + 10, x + bw), max(y + 10, y + bh))))
    return crops


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
    compile_warmup_sec: float,
    initialization_sec: float,
    motion_scan_sec: float,
    chunk_prep_sec: float,
    propagation_sec: float,
    transfer_sec: float,
    geometry_sec: float,
    queue_wait_sec: float,
    discovery_sec: float,
    classification_sec: float,
    persistence_sec: float,
    export_deliverables_sec: float,
    export_mp4_sec: float,
    unique_colonies: int,
    keyframes_count: int,
    committed_frames: int,
) -> Dict[str, Any]:
    """Builds the per-stage timing report (seconds and percent of elapsed) stored on the job."""
    decode_prep = motion_scan_sec + chunk_prep_sec
    critical_path = (
        decode_prep + propagation_sec + transfer_sec + geometry_sec
        + discovery_sec + persistence_sec + export_deliverables_sec + export_mp4_sec
    )
    idle = max(0.0, elapsed_total - critical_path)
    denom = max(elapsed_total, 0.001)

    def pct(v: float) -> float:
        return round((v / denom) * 100.0, 1)

    return {
        "total_elapsed_sec": round(elapsed_total, 3),
        "critical_path_elapsed_sec": round(elapsed_total, 3),
        "compile_warmup_sec": round(compile_warmup_sec, 3),
        "initialization_sec": round(initialization_sec, 3),
        "video_decode_prep_sec": round(decode_prep, 3),
        "motion_scan_sec": round(motion_scan_sec, 3),
        "chunk_prep_sec": round(chunk_prep_sec, 3),
        "synchronized_gpu_inference_sec": round(propagation_sec, 3),
        "mask_transfer_sec": round(transfer_sec, 3),
        "geometry_extraction_sec": round(geometry_sec, 3),
        "queue_wait_sec": round(queue_wait_sec, 3),
        "keyframe_discovery_sec": round(discovery_sec, 3),
        "background_classification_sec": round(classification_sec, 3),
        "keyframe_classification_sec": round(classification_sec, 3),
        "persistence_sec": round(persistence_sec, 3),
        "export_deliverables_sec": round(export_deliverables_sec, 3),
        "export_annotated_mp4_sec": round(export_mp4_sec, 3),
        "unaccounted_idle_sec": round(idle, 3),
        "pct_gpu_inference": pct(propagation_sec),
        "pct_video_decode_prep": pct(decode_prep),
        "pct_keyframe_discovery": pct(discovery_sec),
        "pct_geometry_and_transfer": pct(geometry_sec + transfer_sec),
        "pct_persistence": pct(persistence_sec),
        "pct_export": pct(export_deliverables_sec + export_mp4_sec),
        "pct_unaccounted_idle": pct(idle),
        "unique_colonies_count": unique_colonies,
        "keyframes_count": keyframes_count,
        "committed_frames": committed_frames,
    }
