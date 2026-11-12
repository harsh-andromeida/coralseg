"""
Asynchronous Video Inference Engine and Transect Processor for ReefVision.
Coordinates bounded frame ingestion, ego-motion discovery scheduling, SAM-based benthic tracking,
on-demand BioCLIP/YOLO enrichment, and durable status tracking with SQLite persistence.
Optical-flow inference and automatic fallbacks to it have been completely removed.
"""

import os
import time
import shutil
import queue
import tempfile
import threading
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple
from PIL import Image
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from core.device import get_device_info, get_optimal_device
from core.models import (
    acquire_active_job,
    release_active_job,
    reserve_memory_for_auxiliary_models,
)
from core.video.preprocessor import (
    standardize_video_to_25fps,
    extract_video_frames_to_disk,
    prepare_bounded_chunk_frames,
    cleanup_chunk_dir,
)
from core.video.keyframe_selector import EgoMotionKeyframeSelector
from core.video.tracker import BenthicColonyTracker, CoralTrack
from core.video.sam2_tracker import SAM2BenthicTracker
from core.video.efficienttam_tracker import EfficientTAMBenthicTracker
from core.video.export import export_annotated_video, export_transect_csv, export_coco_video
from core.video.store import VideoJobStore
from core.segmentation import run_segmentation
from core.taxonomy import classify_coral_taxonomy, detect_coral_bleaching


class VideoJobState:
    QUEUED = "queued"
    PREPARING = "preparing"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    ERROR = "error"
    PAUSED = "paused"

    # Compatibility alias
    IDLE = "queued"


class StoreFrameMapAdapter:
    """Lazy dictionary-like adapter exposing VideoJobStore frames for video export."""

    def __init__(self, store: VideoJobStore, job_id: str):
        self.store = store
        self.job_id = job_id

    def get(self, frame_idx: int, default=None):
        f = self.store.get_frame(self.job_id, frame_idx)
        if f and "tracks" in f:
            return f["tracks"]
        return default if default is not None else []

    def __getitem__(self, frame_idx: int):
        return self.get(frame_idx, [])

    def __contains__(self, frame_idx: int):
        return self.store.get_frame(self.job_id, frame_idx) is not None

    def items(self):
        committed = sorted(list(self.store.get_committed_frame_indices(self.job_id)))
        batch_size = 64
        for i in range(0, len(committed), batch_size):
            chunk = committed[i : i + batch_size]
            start_f = chunk[0]
            end_f = chunk[-1]
            frames_map = self.store.get_frames_range(self.job_id, start_f, end_f)
            for f_idx in chunk:
                if f_idx in frames_map:
                    yield f_idx, frames_map[f_idx].get("tracks", [])

    def keys(self):
        return self.store.get_committed_frame_indices(self.job_id)

    def values(self):
        for _, tracks in self.items():
            yield tracks


class VideoProcessor:
    """
    Manages asynchronous underwater video processing pipelines with bounded concurrency,
    strict SAM-only execution, snapshot configuration, durable SQLite storage, and isolated error reporting.
    """

    def __init__(
        self,
        sam_loader: Callable,
        bioclip_loader: Callable,
        bleaching_loader: Callable,
        cache_dir: Optional[str] = None,
    ):
        self.sam_loader = sam_loader
        self.bioclip_loader = bioclip_loader
        self.bleaching_loader = bleaching_loader
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..", "..", "exports", "video")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.store = VideoJobStore(db_path=os.path.join(self.cache_dir, "video_jobs.sqlite3"))
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Bounded job queue: default 1 active video job per accelerator
        self._job_queue = queue.Queue()
        self._queue_worker_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
        self._queue_worker_thread.start()

    def close(self):
        """Cleanly releases storage and worker threads."""
        self.store.close()

    def _process_queue_loop(self):
        """Sequential job coordinator running one active video job at a time."""
        while True:
            try:
                job_id, video_path, opts = self._job_queue.get()
                with self._lock:
                    job = self._jobs.get(job_id)
                    if not job or job.get("cancel_requested"):
                        if job:
                            job["state"] = VideoJobState.CANCELLED
                        self.store.set_job_state(job_id, VideoJobState.CANCELLED)
                        self._job_queue.task_done()
                        continue
                    job["state"] = VideoJobState.PREPARING

                self.store.set_job_state(job_id, VideoJobState.PREPARING)
                acquire_active_job()
                try:
                    self._run_job_pipeline(job_id, video_path, opts)
                finally:
                    release_active_job()
                    self._job_queue.task_done()
            except Exception:
                traceback.print_exc()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves unified job information from durable SQLite store, augmented with active in-memory state."""
        db_job = self.store.get_job(job_id)
        if not db_job:
            with self._lock:
                return self._jobs.get(job_id)

        with self._lock:
            active_info = self._jobs.get(job_id)
            if active_info:
                for k in (
                    "current_stage", "current_model", "current_action", "stage_detail",
                    "stage_progress_pct", "logs", "cancel_requested", "pause_requested",
                ):
                    if k in active_info:
                        db_job[k] = active_info[k]

        db_job["time_series"] = self.store.get_time_series(job_id)
        db_job["frame_tracks_map"] = StoreFrameMapAdapter(self.store, job_id)
        return db_job

    def get_active_job(self) -> Optional[Dict[str, Any]]:
        """Retrieves currently running job, or the most recent video job."""
        with self._lock:
            if self._jobs:
                for j in reversed(list(self._jobs.values())):
                    if j.get("state") in (
                        VideoJobState.QUEUED,
                        VideoJobState.PREPARING,
                        VideoJobState.PROCESSING,
                        VideoJobState.FINALIZING,
                        VideoJobState.PAUSED,
                    ):
                        return self.get_job(j["job_id"])
                sorted_jobs = sorted(self._jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)
                if sorted_jobs:
                    return self.get_job(sorted_jobs[0]["job_id"])

        db_active = self.store.get_active_job()
        if db_active:
            return self.get_job(db_active["job_id"])
        return None

    def probe_video(self, video_path: str) -> Dict[str, Any]:
        """Probes video metadata using OpenCV."""
        if not _HAS_CV2:
            raise RuntimeError("OpenCV is required to process video.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / max(fps, 1.0)
        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "filename": os.path.basename(video_path),
            "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
        }

    def start_job(
        self,
        video_path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Initializes and enqueues a background video processing job."""
        job_id = f"vjob_{int(time.time() * 1000)}"

        # 1. Standardize video to constant 25.0 FPS grid if needed
        storage_dir = os.path.dirname(os.path.abspath(video_path))
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        if base_name.endswith("_25fps"):
            std_25fps_path = os.path.abspath(video_path)
        else:
            std_25fps_path = os.path.join(storage_dir, f"{base_name}_25fps.mp4")

        try:
            if not os.path.isfile(std_25fps_path) or os.path.abspath(video_path) != os.path.abspath(std_25fps_path):
                probe = standardize_video_to_25fps(video_path, std_25fps_path)
                active_video_path = std_25fps_path
            else:
                active_video_path = std_25fps_path
                probe = self.probe_video(std_25fps_path)
        except Exception:
            active_video_path = video_path
            probe = self.probe_video(video_path)

        # 2. Snapshot effective execution options at submission
        opts = {
            "overlap_threshold": 0.75,
            "min_stride": 15,
            "max_stride": 45,
            "points_per_side": 16,
            "iou_match_threshold": 0.30,
            "pred_iou_thresh": 0.50,
            "stability_thresh": 0.50,
            "device_pref": "auto",
            "model_profile": "sam2",
            "model_size": 1024,
            "precision": None,
            "worker_count": 1,
            "export_mp4": True,
            "tracker_mode": "sam2",
        }
        if params:
            opts.update(params)

        # Enforce removal of optical-flow inference mode
        raw_mode = str(opts.get("tracker_mode", "sam2")).lower()
        if raw_mode in ("optical_flow", "flow", "lucas_kanade"):
            raise ValueError(
                "Optical flow inference engine has been removed. "
                "Use 'sam2' (default reference) or 'efficienttam' (fast profile)."
            )

        if opts["min_stride"] > opts["max_stride"]:
            raise ValueError(f"min_stride ({opts['min_stride']}) cannot exceed max_stride ({opts['max_stride']})")

        tracker_profile = opts.get("model_profile", "sam2")

        job_info = {
            "job_id": job_id,
            "video_path": active_video_path,
            "raw_video_path": video_path,
            "filename": probe["filename"],
            "metadata": probe,
            "state": VideoJobState.QUEUED,
            "effective_options": dict(opts),
            "current_frame": 0,
            "total_frames": probe["total_frames"],
            "fps_realtime": 0.0,
            "tracker_fps": 0.0,
            "analysis_fps": 0.0,
            "elapsed_sec": 0.0,
            "progress_pct": 0.0,
            "active_corals_count": 0,
            "total_unique_corals": 0,
            "bleaching_prevalence_pct": 0.0,
            "keyframes_count": 0,
            "keyframe_indices": [],
            "error": None,
            "failed_stage": None,
            "failed_frame_range": None,
            "cancel_requested": False,
            "pause_requested": False,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "export_paths": {},
            "export_status": "pending",
            "current_stage": "Queued",
            "current_model": "SAM 2.1 Video Predictor",
            "current_action": "Waiting in processing queue",
            "stage_detail": f"Video: {probe['total_frames']} frames @ {probe['width']}x{probe['height']}",
            "stage_progress_pct": 0.0,
            "logs": [
                f"[{time.strftime('%H:%M:%S')}] Enqueued transect job on '{probe['filename']}' ({probe['width']}x{probe['height']}, {probe['total_frames']} frames @ 25fps)",
                f"[{time.strftime('%H:%M:%S')}] Profile: {tracker_profile}, size={opts['model_size']}, min_stride={opts['min_stride']}, max_stride={opts['max_stride']}",
            ],
            "tracker_mode": tracker_profile,
        }

        # Persist to durable SQLite database
        self.store.create_job(
            job_id=job_id,
            video_path=active_video_path,
            raw_video_path=video_path,
            filename=probe["filename"],
            metadata=probe,
            effective_options=opts,
            tracker_mode=tracker_profile,
            state=VideoJobState.QUEUED,
            logs=job_info["logs"],
        )

        with self._lock:
            self._jobs[job_id] = job_info

        self._job_queue.put((job_id, active_video_path, opts))
        return job_id

    def pause_job(self, job_id: str) -> bool:
        self.store.request_job_pause(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job["state"] in (VideoJobState.PREPARING, VideoJobState.PROCESSING):
                job["pause_requested"] = True
                job["state"] = VideoJobState.PAUSED
                return True
        return False

    def resume_job(self, job_id: str) -> bool:
        self.store.request_job_resume(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job["state"] == VideoJobState.PAUSED:
                job["pause_requested"] = False
                job["state"] = VideoJobState.PROCESSING
                return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        self.store.request_job_cancel(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["cancel_requested"] = True
                if job["state"] == VideoJobState.QUEUED:
                    job["state"] = VideoJobState.CANCELLED
                else:
                    job["state"] = VideoJobState.CANCELLING
                return True
        return False

    def get_frame_annotation(self, job_id: str, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Retrieves tracking results for a specific frame from durable store."""
        frame = self.store.get_frame(job_id, frame_idx)
        if not frame:
            return None
        return {
            "job_id": job_id,
            "frame_idx": frame_idx,
            "is_keyframe": frame.get("is_keyframe", False),
            "tracks": frame.get("tracks", []),
            "active_count": len(frame.get("tracks", [])),
        }

    def get_frame_annotations_range(
        self,
        job_id: str,
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Returns range of frame annotations from durable store."""
        return self.store.get_frames_range(job_id, start_frame, end_frame)

    def get_frames_delta(
        self,
        job_id: str,
        since_seq: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Delta polling for real-time frontend playback and streaming."""
        return self.store.get_frames_delta(job_id, since_seq, limit)

    def _run_job_pipeline(self, job_id: str, video_path: str, opts: Dict[str, Any]):
        """Main worker execution loop with strict SAM execution and durable SQLite persistence."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return

        with self._lock:
            job["state"] = VideoJobState.PREPARING
            job["started_at"] = time.time()
        self.store.set_job_state(job_id, VideoJobState.PREPARING, started_at=time.time())

        frames_dir: Optional[str] = None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            err_msg = f"Failed to open video file: {video_path}"
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = err_msg
                job["failed_stage"] = "video_demux"
            self.store.set_job_state(job_id, VideoJobState.ERROR, error=err_msg, failed_stage="video_demux")
            return

        total_frames = job["metadata"]["total_frames"]
        width = job["metadata"]["width"]
        height = job["metadata"]["height"]
        video_fps = job["metadata"]["fps"]
        model_profile = opts.get("model_profile", "sam2")
        model_size = int(opts.get("model_size", 1024))
        device_pref = opts.get("device_pref", "auto")
        precision = opts.get("precision")

        # 480p proxy configuration for keyframe segmentation
        target_proxy_h = 480
        if height > target_proxy_h:
            kf_proxy_h = target_proxy_h
            kf_proxy_w = max(16, int(round(width * (target_proxy_h / float(height)))))
            kf_scale_x = width / float(kf_proxy_w)
            kf_scale_y = height / float(kf_proxy_h)
            is_kf_proxy = True
        else:
            kf_proxy_w = width
            kf_proxy_h = height
            kf_scale_x = 1.0
            kf_scale_y = 1.0
            is_kf_proxy = False

        def _set_stage(
            stage: Optional[str] = None,
            model: Optional[str] = None,
            action: Optional[str] = None,
            detail: Optional[str] = None,
            log_entry: Optional[str] = None,
        ):
            with self._lock:
                if stage is not None:
                    job["current_stage"] = stage
                if model is not None:
                    job["current_model"] = model
                if action is not None:
                    job["current_action"] = action
                if detail is not None:
                    job["stage_detail"] = detail
                if log_entry:
                    t_str = time.strftime("%H:%M:%S")
                    job.setdefault("logs", []).append(f"[{t_str}] {log_entry}")
                    if len(job["logs"]) > 100:
                        job["logs"] = job["logs"][-100:]

        # 1. Verify memory budget and load foundation models
        try:
            ok, mem_msg = reserve_memory_for_auxiliary_models(device_pref)
            if not ok:
                _set_stage(log_entry=f"Memory admission warning: {mem_msg}")

            _set_stage(
                stage="Loading foundation models...",
                model="CoralSCOP SAM, BioCLIP, YOLO11n",
                action="Initializing inference sessions",
                log_entry="Loading foundation models into resident memory",
            )
            sam_model = self.sam_loader()
            bioclip_bundle = self.bioclip_loader()
            bleaching_model = self.bleaching_loader()

            sam_prov = sam_model["encoder"].get_providers()[0]
            _set_stage(
                stage="Foundation models loaded",
                detail=f"Inference sessions active ({sam_prov})",
                log_entry=f"Foundation models active ({sam_prov})",
            )
        except Exception as e:
            err_msg = f"Model loading failed: {str(e)}"
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = err_msg
                job["failed_stage"] = "model_load"
            self.store.set_job_state(job_id, VideoJobState.ERROR, error=err_msg, failed_stage="model_load")
            cap.release()
            return

        # 2. Camera Ego-Motion Discovery Scheduling (camera motion estimation ONLY)
        selector = EgoMotionKeyframeSelector(
            overlap_threshold=opts["overlap_threshold"],
            min_stride=opts["min_stride"],
            max_stride=opts["max_stride"],
        )
        tracker = BenthicColonyTracker(
            iou_match_threshold=opts["iou_match_threshold"],
            fps=video_fps,
            grace_period_sec=2.0,
        )

        detected_kfs = [0]
        motion_metas = {}
        scan_cap = cv2.VideoCapture(video_path)
        f_scan = 0

        _set_stage(
            stage="Analyzing camera ego-motion & seafloor overlap...",
            model="Lucas-Kanade Ego-Motion Estimator",
            action="Computing camera translation & benthic scene overlap",
            detail=f"Scanning frame 0/{total_frames}",
        )

        while True:
            if job["cancel_requested"]:
                scan_cap.release()
                cap.release()
                with self._lock:
                    job["state"] = VideoJobState.CANCELLED
                self.store.set_job_state(job_id, VideoJobState.CANCELLED)
                return

            ret, f_bgr = scan_cap.read()
            if not ret or f_bgr is None:
                break
            f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            is_kf, m_meta = selector.evaluate_frame(f_rgb)
            motion_metas[f_scan] = {
                "motion_dx": m_meta.get("motion_dx", 0.0),
                "motion_dy": m_meta.get("motion_dy", 0.0),
                "is_scene_cut": m_meta.get("is_scene_cut", False),
            }
            if is_kf and f_scan > 0:
                detected_kfs.append(f_scan)
            f_scan += 1

            if f_scan % 50 == 0 or f_scan == total_frames:
                _set_stage(
                    stage=f"Analyzing camera motion: {f_scan}/{total_frames} frames ({int(f_scan / max(total_frames, 1) * 100)}%)...",
                    detail=f"Detected {len(detected_kfs)} candidate keyframes",
                )
        scan_cap.release()

        keyframe_indices = sorted(list(set(detected_kfs)))
        _set_stage(
            stage="Motion scan complete",
            detail=f"Found {len(keyframe_indices)} keyframes across {total_frames} frames",
            log_entry=f"Camera motion scan complete: detected {len(keyframe_indices)} keyframes",
        )

        # 3. Setup SAM 2 Video Predictor with bounded temporary frame storage
        frames_dir = os.path.join(tempfile.gettempdir(), f"reefvision_frames_{job_id}_{os.getpid()}")

        sam2_tracker = None
        init_attempts = 0
        while init_attempts < 2:
            try:
                init_attempts += 1
                _set_stage(
                    stage="Preparing video frames for SAM...",
                    model="SAM Video Predictor",
                    action=f"Extracting proxy frames to temporary cache ({init_attempts}/2)",
                )
                _, proxy_meta = extract_video_frames_to_disk(
                    video_path,
                    frames_dir,
                    max_dim=480,
                    jpeg_quality=80,
                )

                _set_stage(
                    stage="Initializing SAM inference state...",
                    model="Meta SAM 2.1 Video Predictor",
                    action="Building temporal memory bank",
                )
                if model_profile == "efficienttam":
                    sam2_tracker = EfficientTAMBenthicTracker(
                        device=device_pref,
                        model_size=model_size,
                        precision=precision,
                        offload_to_cpu=True,
                    )
                else:
                    sam2_tracker = SAM2BenthicTracker(
                        device=device_pref,
                        model_id="facebook/sam2.1-hiera-tiny",
                        model_size=model_size,
                        precision=precision,
                        offload_to_cpu=True,
                    )
                sam2_tracker.init_video(frames_dir)
                _set_stage(
                    stage="SAM Video Tracker ready",
                    log_entry=f"Initialized SAM 2.1 Video Predictor (device={sam2_tracker.device}, precision={sam2_tracker.dtype})",
                )
                break
            except Exception as init_err:
                if init_attempts >= 2:
                    err_msg = f"SAM tracker initialization failed: {str(init_err)}"
                    with self._lock:
                        job["state"] = VideoJobState.ERROR
                        job["error"] = err_msg
                        job["failed_stage"] = "sam_init"
                    self.store.set_job_state(job_id, VideoJobState.ERROR, error=err_msg, failed_stage="sam_init")
                    cap.release()
                    cleanup_chunk_dir(frames_dir)
                    return
                time.sleep(1.0)

        with self._lock:
            job["state"] = VideoJobState.PROCESSING
            job["keyframe_indices"] = list(keyframe_indices)
            job["keyframes_count"] = len(keyframe_indices)
        self.store.set_job_state(job_id, VideoJobState.PROCESSING)

        pipeline_start_t = time.time()
        committed_frames = set()

        def _record_frame_state(cur_f: int, current_active_tracks: List[Any]):
            """Records and commits logical output frame state exactly once into durable store."""
            if cur_f in committed_frames or job["cancel_requested"]:
                return
            committed_frames.add(cur_f)

            bleached_count = sum(1 for t in current_active_tracks if t.bleaching.get("is_bleached", False))
            healthy_count = len(current_active_tracks) - bleached_count
            bleach_pct = (bleached_count / max(len(current_active_tracks), 1)) * 100.0

            genus_counts: Dict[str, int] = {}
            for t in current_active_tracks:
                g = t.taxonomy.get("genus", "Coral")
                genus_counts[g] = genus_counts.get(g, 0) + 1
            dom_genus = max(genus_counts, key=genus_counts.get) if genus_counts else "Coral"

            timestamp_sec = round(cur_f / max(video_fps, 1.0), 2)
            m_meta = motion_metas.get(cur_f, {})
            cov_pct = round(sum(t.area_pct for t in current_active_tracks), 2)

            record = {
                "frame_index": cur_f,
                "timestamp_sec": timestamp_sec,
                "coral_coverage_pct": cov_pct,
                "active_corals_count": len(current_active_tracks),
                "cumulative_unique_corals": tracker.total_unique_colonies,
                "bleached_corals_count": bleached_count,
                "healthy_corals_count": healthy_count,
                "bleaching_prevalence_pct": round(bleach_pct, 1),
                "dominant_genus": dom_genus,
                "is_keyframe": cur_f in keyframe_indices,
                "camera_motion_dx": m_meta.get("motion_dx", 0.0),
                "camera_motion_dy": m_meta.get("motion_dy", 0.0),
            }

            elapsed = max(0.001, time.time() - pipeline_start_t)
            cur_analysis_fps = round(len(committed_frames) / elapsed, 1)

            tracker_timings = sam2_tracker.get_timings()
            prop_sec = tracker_timings.get("propagation_sec", 0.0)
            tracker_fps = round(len(committed_frames) / max(0.001, prop_sec), 1) if prop_sec > 0 else cur_analysis_fps

            # Persist frame geometry and survey metrics into SQLite
            track_snapshots = [t.to_dict() if hasattr(t, "to_dict") else t for t in current_active_tracks]
            self.store.save_frame(
                job_id=job_id,
                frame_idx=cur_f,
                timestamp_sec=timestamp_sec,
                is_keyframe=record["is_keyframe"],
                tracks=track_snapshots,
                metrics=record,
            )

            # Update job progress in SQLite
            self.store.update_job_progress(
                job_id=job_id,
                current_frame=cur_f + 1,
                progress_pct=round((len(committed_frames) / max(total_frames, 1)) * 100.0, 1),
                fps_realtime=cur_analysis_fps,
                tracker_fps=tracker_fps,
                analysis_fps=cur_analysis_fps,
                elapsed_sec=round(elapsed, 2),
                active_corals_count=len(current_active_tracks),
                total_unique_corals=tracker.total_unique_colonies,
                bleaching_prevalence_pct=round(bleach_pct, 1),
                keyframes_count=len(keyframe_indices),
                keyframe_indices=keyframe_indices,
            )

            # Keep in-memory job state bounded (scalars only)
            with self._lock:
                job["current_frame"] = cur_f + 1
                job["progress_pct"] = round((len(committed_frames) / max(total_frames, 1)) * 100.0, 1)
                job["active_corals_count"] = len(current_active_tracks)
                job["total_unique_corals"] = tracker.total_unique_colonies
                job["bleaching_prevalence_pct"] = round(bleach_pct, 1)
                job["analysis_fps"] = cur_analysis_fps
                job["tracker_fps"] = tracker_fps
                job["fps_realtime"] = cur_analysis_fps
                job["elapsed_sec"] = round(elapsed, 2)

        try:
            scale_x = proxy_meta.get("scale_x", 1.0)
            scale_y = proxy_meta.get("scale_y", 1.0)

            # Process between keyframes
            for w_idx in range(len(keyframe_indices)):
                if job["cancel_requested"]:
                    with self._lock:
                        job["state"] = VideoJobState.CANCELLED
                    self.store.set_job_state(job_id, VideoJobState.CANCELLED)
                    break

                while job["pause_requested"] and not job["cancel_requested"]:
                    time.sleep(0.1)

                kf_frame_idx = keyframe_indices[w_idx]
                next_kf_idx = keyframe_indices[w_idx + 1] if w_idx + 1 < len(keyframe_indices) else total_frames
                window_length = next_kf_idx - kf_frame_idx

                # --- 1. KEYFRAME DISCOVERY & CORRECTION STEP ---
                cap.set(cv2.CAP_PROP_POS_FRAMES, kf_frame_idx)
                ret, kf_bgr = cap.read()
                if ret and kf_bgr is not None:
                    kf_rgb = cv2.cvtColor(kf_bgr, cv2.COLOR_BGR2RGB)
                    if is_kf_proxy:
                        kf_proc_bgr = cv2.resize(kf_bgr, (kf_proxy_w, kf_proxy_h), interpolation=cv2.INTER_AREA)
                        kf_proc_rgb = cv2.cvtColor(kf_proc_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        kf_proc_rgb = kf_rgb

                    points_per_side = opts.get("points_per_side", 16)
                    _set_stage(
                        stage=f"Keyframe #{w_idx+1}/{len(keyframe_indices)} (Frame {kf_frame_idx}): Segmenting benthic colonies...",
                        model="CoralSCOP SAM ViT-B",
                        action=f"Prompt grid segmentation ({points_per_side}x{points_per_side} points)",
                        detail=f"Frame {kf_frame_idx} ({kf_proxy_w}x{kf_proxy_h})",
                    )
                    masks_info, _ = run_segmentation(
                        model=sam_model,
                        image=kf_proc_rgb,
                        points_per_side=points_per_side,
                        pred_iou_thresh=opts.get("pred_iou_thresh", 0.50),
                        stability_score_thresh=opts.get("stability_thresh", 0.50),
                        min_mask_region_area=25,
                    )

                    # Remap proxy coordinates to display coordinates
                    if is_kf_proxy:
                        for m in masks_info:
                            if "polygon" in m and m["polygon"]:
                                m["polygon"] = [
                                    [int(round(pt[0] * kf_scale_x)), int(round(pt[1] * kf_scale_y))]
                                    for pt in m["polygon"]
                                ]
                            if "bbox" in m:
                                bx, by, bw, bh = m["bbox"]
                                m["bbox"] = [
                                    int(round(bx * kf_scale_x)),
                                    int(round(by * kf_scale_y)),
                                    max(1, int(round(bw * kf_scale_x))),
                                    max(1, int(round(bh * kf_scale_y))),
                                ]
                            if "centroid" in m:
                                cx, cy = m["centroid"]
                                m["centroid"] = (int(round(cx * kf_scale_x)), int(round(cy * kf_scale_y)))
                            if "area" in m:
                                m["area"] = int(round(m["area"] * kf_scale_x * kf_scale_y))
                            if "area_px" in m:
                                m["area_px"] = int(round(m["area_px"] * kf_scale_x * kf_scale_y))

                    tracker.match_keyframe_detections(
                        new_detections=masks_info,
                        frame_idx=kf_frame_idx,
                        frame_w=width,
                        frame_h=height,
                    )

                    # Enrich newly spawned tracks with BioCLIP and Bleaching
                    new_tracks = tracker.get_tracks_needing_enrichment()
                    if new_tracks:
                        _set_stage(
                            stage=f"Keyframe #{w_idx+1}: Classifying {len(new_tracks)} new corals with BioCLIP & YOLO...",
                            model="BioCLIP ViT & YOLO11n Bleaching",
                            action="Taxonomy and bleaching health inference",
                            detail=f"Extracting {len(new_tracks)} crops",
                        )
                        pil_img = Image.fromarray(kf_rgb)
                        crops = []
                        for trk in new_tracks:
                            x, y, bw, bh = trk.bbox
                            x0 = max(0, x - int(bw * 0.1))
                            y0 = max(0, y - int(bh * 0.1))
                            x1 = min(width, x + bw + int(bw * 0.1))
                            y1 = min(height, y + bh + int(bh * 0.1))
                            if (x1 - x0) >= 10 and (y1 - y0) >= 10:
                                crops.append(pil_img.crop((x0, y0, x1, y1)))
                            else:
                                crops.append(pil_img.crop((x, y, max(x + 10, x + bw), max(y + 10, y + bh))))

                        if crops:
                            tax_results = classify_coral_taxonomy(crops, bioclip_bundle=bioclip_bundle, top_k=3)
                            bleach_results = detect_coral_bleaching(crops, bleaching_model=bleaching_model)

                            for k, trk in enumerate(new_tracks):
                                trk.taxonomy = tax_results[k]
                                trk.bleaching = bleach_results[k]
                                trk.needs_enrichment = False

                                # Save track metadata to SQLite
                                self.store.save_track(
                                    job_id=job_id,
                                    track_id=trk.track_id,
                                    global_colony_id=trk.track_id,
                                    first_frame=kf_frame_idx,
                                    last_frame=kf_frame_idx,
                                    genus=trk.taxonomy.get("genus", "Coral"),
                                    growth_form=trk.taxonomy.get("growth_form", "-"),
                                    taxon_conf=float(trk.taxonomy.get("confidence_pct", 85.0)),
                                    condition=str(trk.bleaching.get("status", "Healthy")),
                                    condition_conf=float(trk.bleaching.get("confidence_pct", 90.0)),
                                    color_rgb=trk.color_rgb,
                                    metadata={"color_hex": trk.color_hex, "predicted_iou": trk.predicted_iou},
                                )

                                # Register newly discovered coral into SAM 2 tracker
                                m_reg = getattr(trk, "mask", None)
                                if m_reg is None and trk.polygon:
                                    m_reg = np.zeros((height, width), dtype=np.uint8)
                                    pts = np.array(trk.polygon, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.fillPoly(m_reg, [pts], 1)
                                    m_reg = m_reg.astype(bool)
                                if m_reg is not None:
                                    try:
                                        sam2_tracker.register_mask(
                                            frame_idx=kf_frame_idx,
                                            obj_id=trk.track_id,
                                            binary_mask=m_reg,
                                            is_correction=False,
                                            metadata={
                                                "track_id": trk.track_id,
                                                "genus": trk.taxonomy.get("genus", "Coral"),
                                                "growth_form": trk.taxonomy.get("growth_form", "-"),
                                                "taxon_conf": trk.taxonomy.get("confidence_pct", 85.0),
                                                "condition": trk.bleaching.get("status", "Healthy"),
                                                "condition_conf": trk.bleaching.get("confidence_pct", 90.0),
                                                "is_bleached": trk.bleaching.get("is_bleached", False),
                                                "color_hex": trk.color_hex,
                                                "first_seen_frame": kf_frame_idx,
                                                "predicted_iou": trk.predicted_iou,
                                            },
                                        )
                                    except Exception as reg_err:
                                        _set_stage(log_entry=f"Warning registering coral {trk.track_id} into SAM: {reg_err}")

                    # Retire absent colonies to relieve active inference memory
                    retired_ids = tracker.retire_absent_tracks(kf_frame_idx)
                    if retired_ids:
                        sam2_tracker.retire_objects(retired_ids)

                # Record keyframe state immediately
                active_tracks = tracker.get_active_tracks()
                _record_frame_state(kf_frame_idx, active_tracks)

                # --- 2. SAM PROPAGATION STREAMING (Intervening Frames) ---
                if window_length > 1:
                    _set_stage(
                        stage=f"Tracking frames {kf_frame_idx+1}-{next_kf_idx-1} via SAM 2...",
                        model="Meta SAM 2.1 Video Predictor",
                        action="Propagating neural deformable masks",
                        detail=f"Streaming {window_length - 1} frames across {sam2_tracker.total_tracks_count} colonies",
                    )
                    prop_attempts = 0
                    while prop_attempts < 2:
                        try:
                            prop_attempts += 1
                            for out_f, det_tracks in sam2_tracker.propagate(
                                start_frame_idx=kf_frame_idx + 1,
                                end_frame_idx=next_kf_idx,
                                scale_x=scale_x,
                                scale_y=scale_y,
                            ):
                                if job["cancel_requested"]:
                                    break
                                for det in det_tracks:
                                    trk_id = det["track_id"]
                                    if trk_id in tracker._tracks:
                                        t = tracker._tracks[trk_id]
                                        t.update_geometry(
                                            bbox=det["bbox"],
                                            polygon=det["polygon"],
                                            centroid=det["centroid"],
                                            area_px=det["area_px"],
                                            area_pct=round(det["area_px"] / max(width * height, 1) * 100.0, 4),
                                            frame_idx=out_f,
                                            predicted_iou=det.get("predicted_iou"),
                                        )

                                active_tracks = tracker.get_active_tracks()
                                _record_frame_state(out_f, active_tracks)
                            break
                        except Exception as prop_err:
                            if prop_attempts >= 2:
                                with self._lock:
                                    job["failed_stage"] = "sam_propagate"
                                    job["failed_frame_range"] = [kf_frame_idx + 1, next_kf_idx - 1]
                                raise RuntimeError(
                                    f"SAM propagation failed on frames {kf_frame_idx+1}-{next_kf_idx-1}: {str(prop_err)}"
                                )
                            time.sleep(0.5)

            # Ensure exact complete frame coverage across entire video
            for f_check in range(total_frames):
                if f_check not in committed_frames and not job["cancel_requested"]:
                    _record_frame_state(f_check, [])

            # Processing loop finished
            if not job["cancel_requested"]:
                with self._lock:
                    job["state"] = VideoJobState.FINALIZING
                self.store.set_job_state(job_id, VideoJobState.FINALIZING)

                _set_stage(
                    stage="Finalizing survey deliverables...",
                    model="Transect Exporter",
                    action="Generating analytics CSV and COCO dataset",
                    detail="Writing survey deliverables to cache",
                )

                transect_csv = os.path.join(self.cache_dir, f"{job_id}_transect.csv")
                coco_json = os.path.join(self.cache_dir, f"{job_id}_coco_video.json")
                annotated_mp4 = os.path.join(self.cache_dir, f"{job_id}_annotated.mp4")

                time_series_data = self.store.get_time_series(job_id)
                frame_adapter = StoreFrameMapAdapter(self.store, job_id)

                try:
                    export_transect_csv(
                        output_csv_path=transect_csv,
                        time_series_records=time_series_data,
                    )
                    export_coco_video(
                        output_json_path=coco_json,
                        video_filename=job["filename"],
                        width=width,
                        height=height,
                        total_frames=total_frames,
                        fps=video_fps,
                        all_tracks=list(tracker._tracks.values()),
                        frame_tracks_map=frame_adapter,
                    )
                except Exception as export_err:
                    _set_stage(log_entry=f"Deliverable export warning: {export_err}")

                # Optional MP4 overlay generation
                if opts.get("export_mp4", True):
                    try:
                        _set_stage(
                            stage="Rendering annotated video overlay...",
                            action="Encoding H.264 preview video",
                        )
                        export_annotated_video(
                            input_video_path=video_path,
                            output_mp4_path=annotated_mp4,
                            frame_tracks_map=frame_adapter,
                        )
                    except Exception as mp4_err:
                        _set_stage(log_entry=f"MP4 export warning: {mp4_err}")

                elapsed_total = round(time.time() - pipeline_start_t, 2)
                _set_stage(
                    stage="Transect analysis completed",
                    model="Completed",
                    action="Ready for playback & review",
                    detail=f"Tracked {tracker.total_unique_colonies} unique corals in {elapsed_total}s ({job['analysis_fps']} FPS)",
                    log_entry=f"Survey finished: {tracker.total_unique_colonies} corals, {job['bleaching_prevalence_pct']}% bleaching across {len(committed_frames)} frames in {elapsed_total}s",
                )

                export_paths = {
                    "annotated_mp4": annotated_mp4 if os.path.exists(annotated_mp4) else None,
                    "transect_csv": transect_csv if os.path.exists(transect_csv) else None,
                    "coco_json": coco_json if os.path.exists(coco_json) else None,
                }

                with self._lock:
                    job["state"] = VideoJobState.COMPLETED
                    job["completed_at"] = time.time()
                    job["export_status"] = "ready"
                    job["export_paths"] = export_paths

                self.store.set_job_state(
                    job_id=job_id,
                    state=VideoJobState.COMPLETED,
                    completed_at=time.time(),
                    export_paths=export_paths,
                    export_status="ready",
                )

        except Exception as e:
            traceback.print_exc()
            err_msg = f"Pipeline execution error: {str(e)}"
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = err_msg
            self.store.set_job_state(
                job_id=job_id,
                state=VideoJobState.ERROR,
                error=err_msg,
                failed_stage=job.get("failed_stage"),
                failed_frame_range=str(job.get("failed_frame_range")),
            )
            _set_stage(
                stage="Pipeline execution halted on error",
                model="Error",
                detail=err_msg,
                log_entry=f"ERROR: {err_msg}",
            )
        finally:
            cap.release()
            if sam2_tracker:
                sam2_tracker.close()
            cleanup_chunk_dir(frames_dir)
