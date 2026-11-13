"""
Video ingestion, real-time tracking, metrics telemetry, and video export router for ReefVision.
"""

import os
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from core.models import check_models_download_status
from core.video import (
    render_video_overlay,
    export_annotated_video,
    export_transect_csv,
    export_coco_video,
)
from core.video.preprocessor import standardize_video_to_25fps
from app.schemas import VideoProcessRequest, VideoStatusResponse
from app import state

router = APIRouter(tags=["Video Processing & Benthic Tracking"])


def _check_models_status() -> tuple[bool, list]:
    """Helper that checks models download status, respecting test monkey patching on app.main."""
    try:
        import app.main as app_main
        checker = getattr(app_main, "check_models_download_status", check_models_download_status)
        return checker()
    except Exception:
        return check_models_download_status()


@router.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads an underwater video (.mp4, .mov, .avi) and probes its metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    allowed_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    clean_name = os.path.basename(file.filename)
    dest_path = os.path.join(state.VIDEO_STORAGE_DIR, clean_name)

    with open(dest_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    base, ext = os.path.splitext(clean_name)
    if base.endswith("_25fps"):
        std_name = clean_name
    else:
        std_name = f"{base}_25fps.mp4"
    std_path = os.path.join(state.VIDEO_STORAGE_DIR, std_name)

    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")

    try:
        metadata = standardize_video_to_25fps(dest_path, std_path)
        final_name = std_name
        final_path = std_path
    except Exception:
        try:
            metadata = state.video_processor.probe_video(dest_path)
            final_name = clean_name
            final_path = dest_path
        except Exception as probe_err:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise HTTPException(status_code=400, detail=f"Failed to probe video: {str(probe_err)}")

    return {
        "status": "success",
        "video_name": final_name,
        "video_path": final_path,
        "metadata": metadata,
    }


@router.get("/api/video/sample")
def get_sample_video():
    """Loads a pre-packaged underwater coral reef transect video standardized to 25 FPS."""
    sample_25fps = os.path.join(state.BACKEND_DIR.parent, "demo_images", "sample_reef_transect_25fps.mp4")
    sample_src = os.path.join(state.BACKEND_DIR.parent, "demo_images", "sample_reef_transect.mp4")

    clean_name = "sample_reef_transect_25fps.mp4"
    dest_path = os.path.join(state.VIDEO_STORAGE_DIR, clean_name)

    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")

    if os.path.isfile(sample_25fps):
        if not os.path.isfile(dest_path) or os.path.getmtime(sample_25fps) > os.path.getmtime(dest_path):
            shutil.copyfile(sample_25fps, dest_path)
        metadata = state.video_processor.probe_video(dest_path)
    elif os.path.isfile(sample_src):
        if not os.path.isfile(dest_path) or os.path.getmtime(sample_src) > os.path.getmtime(dest_path):
            try:
                metadata = standardize_video_to_25fps(sample_src, dest_path)
            except Exception:
                shutil.copyfile(sample_src, dest_path)
                metadata = state.video_processor.probe_video(dest_path)
        else:
            metadata = state.video_processor.probe_video(dest_path)
    else:
        raise HTTPException(status_code=404, detail="Sample video not found in demo_images directory.")

    return {
        "status": "success",
        "video_name": clean_name,
        "video_path": dest_path,
        "metadata": metadata,
    }


@router.get("/api/video/stream-file/{video_name}")
def stream_video_file(video_name: str):
    """Streams raw video file for HTML5 video player playback with HTTP range support."""
    clean_base = os.path.basename(video_name)
    target = os.path.join(state.VIDEO_STORAGE_DIR, clean_base)
    if not os.path.isfile(target):
        # Gracefully handle redundant _25fps_25fps naming
        if "_25fps_25fps" in clean_base:
            normalized_name = clean_base.replace("_25fps_25fps", "_25fps")
            candidate = os.path.join(state.VIDEO_STORAGE_DIR, normalized_name)
            if os.path.isfile(candidate):
                target = candidate

    if not os.path.isfile(target):
        demo_target = os.path.join(state.BACKEND_DIR.parent, "demo_images", clean_base)
        if os.path.isfile(demo_target):
            target = demo_target
        elif "_25fps_25fps" in clean_base:
            demo_norm = os.path.join(state.BACKEND_DIR.parent, "demo_images", clean_base.replace("_25fps_25fps", "_25fps"))
            if os.path.isfile(demo_norm):
                target = demo_norm

    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"Video file '{video_name}' not found.")

    return FileResponse(
        target,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.post("/api/video/process")
def process_video(req: VideoProcessRequest):
    """Starts asynchronous high-FPS benthic tracking and keyframe analysis on a video."""
    all_ready, _ = _check_models_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are not downloaded. Please download all models before running video processing."
        )

    candidate_path = os.path.join(state.VIDEO_STORAGE_DIR, req.video_name)
    if not os.path.isfile(candidate_path):
        if os.path.isfile(req.video_name):
            candidate_path = req.video_name
        else:
            raise HTTPException(status_code=404, detail=f"Video file '{req.video_name}' not found.")

    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")

    try:
        job_id = state.video_processor.start_job(
            video_path=candidate_path,
            params={
                "overlap_threshold": req.overlap_threshold,
                "min_stride": req.min_stride,
                "max_stride": req.max_stride,
                "points_per_side": req.points_per_side,
                "iou_match_threshold": req.iou_match_threshold,
                "pred_iou_thresh": req.pred_iou_thresh,
                "stability_thresh": req.stability_thresh,
                "device_pref": state._ACTIVE_DEVICE_PREF,
                "tracker_mode": req.tracker_mode or "sam2",
                "model_profile": req.model_profile,
                "model_size": req.model_size,
                "precision": req.precision,
                "worker_count": req.worker_count,
                "compile_tracker": req.compile_tracker,
                "export_mp4": req.export_mp4,
            },
        )
        job = state.video_processor.get_job(job_id)
        return {
            "status": "started",
            "job_id": job_id,
            "filename": job["filename"],
            "metadata": job["metadata"],
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start video processing: {str(e)}")


@router.get("/api/video/active-status", response_model=Optional[VideoStatusResponse])
def get_active_video_status():
    """Retrieves real-time processing status of the active or most recent video job."""
    if not state.video_processor:
        return None
    job = state.video_processor.get_active_job()
    if not job:
        return None

    return VideoStatusResponse(
        job_id=job["job_id"],
        filename=job["filename"],
        state=job["state"],
        current_frame=job["current_frame"],
        total_frames=job["total_frames"],
        progress_pct=job["progress_pct"],
        fps_realtime=job.get("fps_realtime", 0.0),
        tracker_fps=job.get("tracker_fps", 0.0),
        analysis_fps=job.get("analysis_fps", 0.0),
        elapsed_sec=job.get("elapsed_sec", 0.0),
        active_corals_count=job["active_corals_count"],
        total_unique_corals=job["total_unique_corals"],
        bleaching_prevalence_pct=job["bleaching_prevalence_pct"],
        keyframes_count=job["keyframes_count"],
        metadata=job["metadata"],
        error=job["error"],
        time_series=job["time_series"],
        export_paths=job.get("export_paths"),
        export_status=job.get("export_status", "pending"),
        current_stage=job.get("current_stage", "Processing"),
        current_model=job.get("current_model"),
        current_action=job.get("current_action"),
        stage_detail=job.get("stage_detail"),
        logs=job.get("logs", []),
        tracker_mode=job.get("tracker_mode", "sam2"),
    )


@router.get("/api/video/status/{job_id}", response_model=VideoStatusResponse)
def get_video_status(job_id: str, include_metrics: bool = False):
    """Retrieves real-time processing status, FPS, and optional time-series survey telemetry."""
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    job = state.video_processor.get_job(job_id, include_time_series=include_metrics)
    if not job:
        raise HTTPException(status_code=404, detail=f"Video job '{job_id}' not found.")

    return VideoStatusResponse(
        job_id=job["job_id"],
        filename=job["filename"],
        state=job["state"],
        current_frame=job["current_frame"],
        total_frames=job["total_frames"],
        progress_pct=job["progress_pct"],
        fps_realtime=job.get("fps_realtime", 0.0),
        tracker_fps=job.get("tracker_fps", 0.0),
        analysis_fps=job.get("analysis_fps", 0.0),
        elapsed_sec=job.get("elapsed_sec", 0.0),
        active_corals_count=job["active_corals_count"],
        total_unique_corals=job["total_unique_corals"],
        bleaching_prevalence_pct=job["bleaching_prevalence_pct"],
        keyframes_count=job["keyframes_count"],
        metadata=job["metadata"],
        error=job["error"],
        time_series=job.get("time_series", []),
        export_paths=job.get("export_paths"),
        export_status=job.get("export_status", "pending"),
        current_stage=job.get("current_stage", "Processing"),
        current_model=job.get("current_model"),
        current_action=job.get("current_action"),
        stage_detail=job.get("stage_detail"),
        logs=job.get("logs", []),
        tracker_mode=job.get("tracker_mode", "sam2"),
    )


@router.get("/api/video/metrics/{job_id}")
def get_video_metrics(job_id: str):
    """Returns full survey telemetry time-series for analysis and charting without polluting status polls."""
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    job = state.video_processor.get_job(job_id, include_time_series=True)
    if not job:
        raise HTTPException(status_code=404, detail=f"Video job '{job_id}' not found.")
    return {
        "job_id": job_id,
        "metrics_count": len(job.get("time_series", [])),
        "time_series": job.get("time_series", []),
    }


@router.post("/api/video/pause/{job_id}")
def pause_video_job(job_id: str):
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    success = state.video_processor.pause_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not pause job or job not running.")
    return {"status": "paused", "job_id": job_id}


@router.post("/api/video/resume/{job_id}")
def resume_video_job(job_id: str):
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    success = state.video_processor.resume_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not resume job or job not paused.")
    return {"status": "resumed", "job_id": job_id}


@router.post("/api/video/cancel/{job_id}")
def cancel_video_job(job_id: str):
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    success = state.video_processor.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"status": "cancelled", "job_id": job_id}


@router.get("/api/video/frame/{job_id}/{frame_idx}")
def get_video_frame_annotations(job_id: str, frame_idx: int):
    """Returns tracking detections and polygon boundaries for a specific frame."""
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    data = state.video_processor.get_frame_annotation(job_id, frame_idx)
    if data is None:
        raise HTTPException(status_code=404, detail="Frame data not available.")
    return data


@router.get("/api/video/frames/{job_id}")
def get_video_all_frame_annotations(
    job_id: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
):
    """
    Returns tracking detections and polygon boundaries for a range or all processed frames.
    Allows frontend to pre-cache frame annotations in bulk for 60 FPS stutter-free playback.
    """
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    job = state.video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")

    frames_map = state.video_processor.get_frame_annotations_range(job_id, start_frame, end_frame)
    keyframes = set(job.get("keyframe_indices", []))

    result = {}
    for f_idx, f_data in frames_map.items():
        raw_tracks = f_data.get("tracks", [])
        result[f_idx] = {
            "frame_idx": f_idx,
            "is_keyframe": f_idx in keyframes or bool(f_data.get("is_keyframe", False)),
            "tracks": [t.to_dict() if hasattr(t, "to_dict") else t for t in raw_tracks],
        }

    return {
        "job_id": job_id,
        "total_frames_available": len(result),
        "frames": result,
    }


@router.get("/api/video/delta/{job_id}")
def get_video_frames_delta(
    job_id: str,
    since_seq: int = 0,
    limit: int = 100,
):
    """
    Returns frame annotations committed since sequence `since_seq`.
    Provides low-overhead streaming updates for real-time frontend playback.
    """
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    job = state.video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")

    frames, max_seq = state.video_processor.get_frames_delta(job_id, since_seq, limit)
    return {
        "job_id": job_id,
        "since_seq": since_seq,
        "max_seq": max_seq,
        "count": len(frames),
        "frames": frames,
    }


@router.get("/api/video/export/{job_id}")
def export_video_result(job_id: str, format: str = "mp4"):
    """Downloads annotated MP4, Transect CSV, or MOT/COCO JSON with on-demand fallback."""
    if not state.video_processor:
        raise HTTPException(status_code=500, detail="VideoProcessor is not initialized.")
    job = state.video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")

    export_paths = job.get("export_paths", {})
    fmt = format.lower().strip()
    cache_dir = state.video_processor.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    if fmt == "mp4":
        p = export_paths.get("annotated_mp4")
        if not p or not os.path.isfile(p):
            candidate = os.path.join(cache_dir, f"{job_id}_annotated.mp4")
            if os.path.isfile(candidate):
                p = candidate
                job.setdefault("export_paths", {})["annotated_mp4"] = candidate
            elif job.get("frame_tracks_map") and job.get("video_path") and os.path.isfile(job["video_path"]):
                try:
                    export_annotated_video(
                        input_video_path=job["video_path"],
                        output_mp4_path=candidate,
                        frame_tracks_map=job["frame_tracks_map"],
                        max_frames=job.get("current_frame"),
                    )
                    if os.path.isfile(candidate):
                        p = candidate
                        job.setdefault("export_paths", {})["annotated_mp4"] = candidate
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to generate annotated video: {str(e)}")

        if not p or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="Annotated MP4 export not ready. Please wait for frames to process.")
        return FileResponse(p, media_type="video/mp4", filename=os.path.basename(p))

    elif fmt == "csv":
        p = export_paths.get("transect_csv")
        if not p or not os.path.isfile(p):
            candidate = os.path.join(cache_dir, f"{job_id}_transect.csv")
            if os.path.isfile(candidate):
                p = candidate
                job.setdefault("export_paths", {})["transect_csv"] = candidate
            else:
                try:
                    ts_records = state.video_processor.store.get_time_series(job_id)
                    if ts_records:
                        export_transect_csv(
                            output_csv_path=candidate,
                            time_series_records=ts_records,
                        )
                        if os.path.isfile(candidate):
                            p = candidate
                            job.setdefault("export_paths", {})["transect_csv"] = candidate
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

        if not p or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="Transect CSV export not ready.")
        return FileResponse(p, media_type="text/csv", filename=os.path.basename(p))

    elif fmt in ("json", "coco", "mot"):
        p = export_paths.get("coco_json")
        if not p or not os.path.isfile(p):
            candidate = os.path.join(cache_dir, f"{job_id}_coco_video.json")
            if os.path.isfile(candidate):
                p = candidate
                job.setdefault("export_paths", {})["coco_json"] = candidate
            elif job.get("frame_tracks_map"):
                try:
                    meta = job.get("metadata", {})
                    all_tracks_dict = {}
                    for f_tracks in job["frame_tracks_map"].values():
                        for tr in f_tracks:
                            tid = getattr(tr, "track_id", None) or (tr.get("track_id") if isinstance(tr, dict) else 0)
                            all_tracks_dict[tid] = tr

                    export_coco_video(
                        output_json_path=candidate,
                        video_filename=job.get("filename", "video.mp4"),
                        width=meta.get("width", 1920),
                        height=meta.get("height", 1080),
                        total_frames=meta.get("total_frames", job.get("total_frames", 1)),
                        fps=meta.get("fps", 25.0),
                        all_tracks=list(all_tracks_dict.values()),
                        frame_tracks_map=job["frame_tracks_map"],
                    )
                    if os.path.isfile(candidate):
                        p = candidate
                        job.setdefault("export_paths", {})["coco_json"] = candidate
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to generate COCO JSON: {str(e)}")

        if not p or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="COCO JSON export not ready.")
        return FileResponse(p, media_type="application/json", filename=os.path.basename(p))
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: mp4, csv, json")
