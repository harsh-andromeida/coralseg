"""
Asynchronous Video Inference Engine and Transect Processor for ReefVision.
Coordinates bounded frame ingestion, ego-motion discovery scheduling, SAM-based benthic tracking,
on-demand BioCLIP/YOLO enrichment, and durable status tracking with SQLite persistence.
All temporal mask propagation is driven by spatio-temporal memory attention.
"""

import os
import time
import shutil
import queue
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
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
    BoundedVideoFrameBuffer,
)
from core.video.keyframe_selector import EgoMotionKeyframeSelector
from core.video.tracker import BenthicColonyTracker, CoralTrack
from core.video.sam2_tracker import SAM2BenthicTracker
from core.video.efficienttam_tracker import EfficientTAMBenthicTracker
from core.video.scheduler import calculate_chunk_plan, admit_workers_for_device, reconcile_chunk_identities
from core.video.store import VideoJobStore
from core.segmentation import run_segmentation
from core.taxonomy import classify_coral_taxonomy, detect_coral_bleaching
from core.video.pipeline_stages import (
    build_pipeline_timings,
    crop_track_regions,
    rescale_detections_to_full_res,
    run_deliverable_exports,
    track_metadata,
)


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
        self._tracker_cache: Dict[Tuple[str, int, str, Optional[str], bool], Any] = {}
        self._recover_orphaned_jobs()

        # Bounded job queue: default 1 active video job per accelerator
        self._job_queue = queue.Queue()
        self._queue_worker_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
        self._queue_worker_thread.start()

    def close(self):
        """Cleanly releases storage, cached tracker models, and worker threads."""
        self.store.close()
        with self._lock:
            for trk in self._tracker_cache.values():
                try:
                    trk.close()
                except Exception:
                    pass
            self._tracker_cache.clear()

    _TERMINAL_STATES = (VideoJobState.COMPLETED, VideoJobState.ERROR, VideoJobState.CANCELLED)

    def _recover_orphaned_jobs(self):
        """Marks jobs left running by a previous backend process as interrupted.

        Their worker thread died with that process, so nothing can ever finish or cancel them;
        left as 'processing' they appear active forever and the UI cancel request has no effect.
        """
        try:
            for j in self.store.list_jobs(limit=500):
                if j["state"] not in self._TERMINAL_STATES:
                    self.store.set_job_state(
                        j["job_id"],
                        VideoJobState.ERROR,
                        error="Interrupted: the backend restarted before this job finished.",
                        failed_stage="interrupted",
                    )
        except Exception:
            traceback.print_exc()

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

    def get_job(self, job_id: str, include_time_series: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieves unified job information from durable SQLite store, augmented with active in-memory state.
        By default, excludes heavy time_series history to keep status and delta queries lightweight.
        """
        db_job = self.store.get_job(job_id)
        if not db_job:
            with self._lock:
                return self._jobs.get(job_id)

        with self._lock:
            active_info = self._jobs.get(job_id)
            if active_info:
                for k in (
                    "state", "current_stage", "current_model", "current_action", "stage_detail",
                    "stage_progress_pct", "logs", "cancel_requested", "pause_requested",
                ):
                    if k in active_info:
                        db_job[k] = active_info[k]

        if include_time_series:
            db_job["time_series"] = self.store.get_time_series(job_id)
        else:
            db_job["time_series"] = []

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
            "worker_count": "auto",
            "compile_tracker": False,
            "export_mp4": True,
            "tracker_mode": "sam2",
        }
        if params:
            opts.update(params)

        # Validate tracker mode
        raw_mode = str(opts.get("tracker_mode", "sam2")).lower()
        if raw_mode not in ("sam2", "efficienttam"):
            raise ValueError(f"Unsupported tracker mode '{raw_mode}'. Allowed: 'sam2', 'efficienttam'")

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
        """Requests cancellation. Works for live jobs and for orphaned jobs known only to the store."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["cancel_requested"] = True
                state = job["state"]
                if state not in self._TERMINAL_STATES:
                    state = VideoJobState.CANCELLED if state == VideoJobState.QUEUED else VideoJobState.CANCELLING
                    job["state"] = state
                new_state = state
            else:
                db_job = self.store.get_job(job_id)
                if db_job is None:
                    return False
                new_state = db_job["state"]
                if new_state not in self._TERMINAL_STATES:
                    # No worker owns this job (e.g. left over from a previous backend run)
                    new_state = VideoJobState.CANCELLED
        self.store.request_job_cancel(job_id)
        # Persist so status polling reflects the request immediately
        self.store.set_job_state(job_id, new_state)
        return True

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

        frame_buffer: Optional[BoundedVideoFrameBuffer] = None
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
                    formatted_log = f"[{t_str}] {log_entry}"
                    job.setdefault("logs", []).append(formatted_log)
                    if len(job["logs"]) > 100:
                        job["logs"] = job["logs"][-100:]
                    print(f"  ⚡ [Pipeline] {log_entry}", flush=True)

        def _finish_cancelled():
            with self._lock:
                job["state"] = VideoJobState.CANCELLED
            self.store.set_job_state(job_id, VideoJobState.CANCELLED)
            _set_stage(
                stage="Analysis cancelled",
                model="Cancelled",
                action="Stopped by user",
                log_entry="Video analysis cancelled by user",
            )

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
            print(f"  ❌ [Pipeline Error] {err_msg}", flush=True)
            traceback.print_exc()
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = err_msg
                job["failed_stage"] = "model_load"
            self.store.set_job_state(job_id, VideoJobState.ERROR, error=err_msg, failed_stage="model_load")
            cap.release()
            return

        if job["cancel_requested"]:
            _finish_cancelled()
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

        # 3. Worker Admission and Bounded Temporal Chunk Planning
        from core.video.scheduler import normalize_worker_count
        requested_workers = opts.get("worker_count", "auto")
        normalized_workers = normalize_worker_count(requested_workers, device_pref=device_pref)
        effective_workers = admit_workers_for_device(device_pref, normalized_workers)
        chunk_size = opts.get("chunk_size", 128)
        overlap_frames = opts.get("overlap_frames", 16)
        chunk_plan = calculate_chunk_plan(total_frames, chunk_size=chunk_size, overlap_frames=overlap_frames)

        detected_kfs = [0]
        motion_metas = {}
        keyframe_indices = set(detected_kfs)
        t_motion_scan_sec = 0.0
        t_chunk_prep_sec = 0.0
        t_discovery_sec = 0.0
        t_classification_sec = 0.0
        t_persistence_sec = 0.0
        t_export_deliv_sec = 0.0
        t_export_mp4_sec = 0.0
        accumulated_transfer_sec = 0.0
        accumulated_geometry_sec = 0.0
        accumulated_init_sec = 0.0

        with self._lock:
            job["worker_count"] = effective_workers
            job["effective_workers"] = effective_workers
            job["keyframe_indices"] = list(keyframe_indices)
            job["keyframes_count"] = len(keyframe_indices)
        self.store.register_chunks(job_id, [(c.chunk_idx, c.start_frame, c.end_frame) for c in chunk_plan])

        _set_stage(
            stage="Planning bounded temporal chunk processing...",
            model="Temporal Chunk Scheduler",
            action=f"Partitioned video into {len(chunk_plan)} chunk(s) (chunk_size={chunk_size}, overlap={overlap_frames})",
            detail=f"{effective_workers} worker(s) admitted for device '{device_pref}'",
            log_entry=f"Chunk plan ready: {len(chunk_plan)} chunk(s) of {chunk_size} frames ({overlap_frames} overlap), {effective_workers} worker(s)",
        )

        compile_tracker = bool(opts.get("compile_tracker", False))
        cache_key = (model_profile, model_size, device_pref, precision, compile_tracker)
        sam2_tracker = None
        with self._lock:
            if cache_key in self._tracker_cache:
                sam2_tracker = self._tracker_cache[cache_key]

        try:
            if sam2_tracker is None:
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
                        compile_tracker=compile_tracker,
                        offload_to_cpu=True,
                    )
                sam2_tracker.load()
                with self._lock:
                    self._tracker_cache[cache_key] = sam2_tracker
            else:
                # Safe model reuse: reset temporal tracking state while retaining compiled weights
                sam2_tracker.reset()
        except Exception as init_err:
            err_msg = f"SAM tracker initialization failed: {str(init_err)}"
            print(f"  ❌ [Pipeline Error] {err_msg}", flush=True)
            traceback.print_exc()
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = err_msg
                job["failed_stage"] = "sam_init"
            self.store.set_job_state(job_id, VideoJobState.ERROR, error=err_msg, failed_stage="sam_init")
            cap.release()
            return

        if job["cancel_requested"]:
            _finish_cancelled()
            cap.release()
            return

        with self._lock:
            job["state"] = VideoJobState.PROCESSING
        self.store.set_job_state(job_id, VideoJobState.PROCESSING)

        # Capture initial load & compilation warm-up timings
        init_timings = sam2_tracker.get_timings()
        compile_warmup_sec = init_timings.get("compile_warmup_sec", 0.0)
        accumulated_init_sec = init_timings.get("initialization_sec", 0.0)
        accumulated_queue_wait_sec = 0.0

        pipeline_start_t = time.time()
        committed_frames = set()
        accumulated_prop_sec = 0.0
        accumulated_prop_frames = 0
        previous_chunk_overlap_frames: Dict[int, List[Dict[str, Any]]] = {}
        overlap_start_frames = {c.start_frame for c in chunk_plan if c.chunk_idx > 0}
        saved_overlap_start_tracks: Dict[int, List[Dict[str, Any]]] = {}

        def _save_overlap_snapshot(f_idx: int, tracks: List[CoralTrack]):
            if f_idx not in overlap_start_frames:
                return
            snaps = []
            for trk in tracks:
                if getattr(trk, "active", False):
                    m_poly = None
                    if trk.polygon:
                        m_poly = np.zeros((height, width), dtype=np.uint8)
                        pts = np.array(trk.polygon, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(m_poly, [pts], 1)
                        m_poly = m_poly.astype(bool)
                    snaps.append({
                        "track_id": trk.track_id,
                        "global_colony_id": trk.global_colony_id,
                        "mask": m_poly,
                        "polygon": trk.polygon,
                        "bbox": trk.bbox,
                        "metadata": {
                            "track_id": trk.track_id,
                            "global_colony_id": trk.global_colony_id,
                            "genus": trk.taxonomy.get("genus", "Coral"),
                            "growth_form": trk.taxonomy.get("growth_form", "-"),
                            "taxon_conf": trk.taxonomy.get("confidence_pct", 85.0),
                            "condition": trk.bleaching.get("status", "Healthy"),
                            "condition_conf": trk.bleaching.get("confidence_pct", 90.0),
                            "is_bleached": trk.bleaching.get("is_bleached", False),
                            "color_hex": trk.color_hex,
                            "first_seen_frame": trk.first_seen_frame,
                            "predicted_iou": trk.predicted_iou,
                        },
                    })
            saved_overlap_start_tracks[f_idx] = snaps

        pending_frame_writes: List[Dict[str, Any]] = []
        last_progress_update_t = 0.0

        def _flush_pending_frames():
            nonlocal t_persistence_sec
            if pending_frame_writes:
                t_pers_start = time.perf_counter()
                self.store.save_frames_batch(job_id=job_id, frames_data=pending_frame_writes)
                t_persistence_sec += (time.perf_counter() - t_pers_start)
                pending_frame_writes.clear()

        def _record_frame_state(cur_f: int, current_active_tracks: List[Any]):
            """Records and commits logical output frame state exactly once into durable store."""
            if cur_f in committed_frames or job["cancel_requested"]:
                return
            committed_frames.add(cur_f)

            bleached_count = sum(1 for t in current_active_tracks if getattr(t, "bleaching", {}).get("is_bleached", False))
            healthy_count = len(current_active_tracks) - bleached_count
            bleach_pct = (bleached_count / max(len(current_active_tracks), 1)) * 100.0

            genus_counts: Dict[str, int] = {}
            for t in current_active_tracks:
                tax = getattr(t, "taxonomy", {}) if hasattr(t, "taxonomy") else {}
                g = tax.get("genus", "Coral")
                genus_counts[g] = genus_counts.get(g, 0) + 1
            dom_genus = max(genus_counts, key=genus_counts.get) if genus_counts else "Coral"

            timestamp_sec = round(cur_f / max(video_fps, 1.0), 2)
            m_meta = motion_metas.get(cur_f, {})
            cov_pct = round(sum(getattr(t, "area_pct", 0.0) for t in current_active_tracks), 2)

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
            cur_prop_sec = accumulated_prop_sec + tracker_timings.get("propagation_sec", 0.0)
            cur_prop_frames = accumulated_prop_frames + tracker_timings.get("frames_propagated", 0)
            tracker_fps = round(cur_prop_frames / max(0.001, cur_prop_sec), 1) if cur_prop_sec > 0 else cur_analysis_fps

            # Persist frame geometry and survey metrics into SQLite in batches
            track_snapshots = [t.to_dict() if hasattr(t, "to_dict") else t for t in current_active_tracks]
            pending_frame_writes.append({
                "frame_idx": cur_f,
                "timestamp_sec": timestamp_sec,
                "is_keyframe": record["is_keyframe"],
                "tracks": track_snapshots,
                "metrics": record,
            })

            # Flush batch every 32 frames, on keyframes, or on final frame
            if len(pending_frame_writes) >= 32 or record["is_keyframe"] or cur_f == total_frames - 1:
                _flush_pending_frames()

            # Throttle SQLite progress writes (at most 4Hz, on keyframes, or on final frame)
            nonlocal last_progress_update_t
            now_t = time.time()
            if (now_t - last_progress_update_t >= 0.25) or record["is_keyframe"] or cur_f == total_frames - 1:
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
                    keyframe_indices=sorted(list(keyframe_indices)),
                )
                last_progress_update_t = now_t

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

        prefetch_executor = ThreadPoolExecutor(max_workers=1)
        enrichment_executor = ThreadPoolExecutor(max_workers=1)
        pending_enrichment_futures: List[Any] = []

        def _reap_enrichment_results(block: bool = False):
            nonlocal t_classification_sec
            still_pending = []
            for fut in pending_enrichment_futures:
                if block or fut.done():
                    try:
                        res_pack = fut.result()
                        t_classification_sec += res_pack.get("duration_sec", 0.0)
                        for trk_id, res in res_pack.get("results", {}).items():
                            # 1. Update live tracker mutable state in coordinator
                            if trk_id in tracker._tracks:
                                trk_obj = tracker._tracks[trk_id]
                                trk_obj.taxonomy = res["taxonomy"]
                                trk_obj.bleaching = res["bleaching"]
                                trk_obj.needs_enrichment = False
                            # 2. Update sam2_tracker metadata so future frames carry finalized labels
                            if hasattr(sam2_tracker, "_tracks_meta") and trk_id in sam2_tracker._tracks_meta:
                                meta = sam2_tracker._tracks_meta[trk_id]
                                meta["genus"] = res["taxonomy"].get("genus", "Coral")
                                meta["growth_form"] = res["taxonomy"].get("growth_form", "-")
                                meta["taxon_conf"] = res["taxonomy"].get("confidence_pct", 85.0)
                                meta["condition"] = res["bleaching"].get("status", "Healthy")
                                meta["condition_conf"] = res["bleaching"].get("confidence_pct", 90.0)
                                meta["is_bleached"] = res["bleaching"].get("is_bleached", False)
                            # 3. Update store track record and finalize classifications
                            self.store.save_track(
                                job_id=job_id,
                                track_id=trk_id,
                                global_colony_id=trk_id,
                                first_frame=res["first_frame"],
                                last_frame=res["first_frame"],
                                genus=res["taxonomy"].get("genus", "Coral"),
                                growth_form=res["taxonomy"].get("growth_form", "-"),
                                taxon_conf=float(res["taxonomy"].get("confidence_pct", 85.0)),
                                condition=str(res["bleaching"].get("status", "Healthy")),
                                condition_conf=float(res["bleaching"].get("confidence_pct", 90.0)),
                                color_rgb=res["color_rgb"],
                                metadata={"color_hex": res["color_hex"], "predicted_iou": res["predicted_iou"]},
                            )
                            self.store.finalize_track_classification(
                                job_id=job_id,
                                track_id=trk_id,
                                taxonomy=res["taxonomy"],
                                bleaching=res["bleaching"],
                            )
                    except Exception as ef:
                        _set_stage(log_entry=f"Async classification task error: {ef}")
                else:
                    still_pending.append(fut)
            pending_enrichment_futures.clear()
            pending_enrichment_futures.extend(still_pending)

        frame_buffer = BoundedVideoFrameBuffer(video_path=video_path, max_dim=480)
        try:
            for chunk_idx, chunk in enumerate(chunk_plan):
                if job["cancel_requested"]:
                    with self._lock:
                        job["state"] = VideoJobState.CANCELLED
                    self.store.set_job_state(job_id, VideoJobState.CANCELLED)
                    break

                while job["pause_requested"] and not job["cancel_requested"]:
                    time.sleep(0.1)

                _set_stage(
                    stage=f"Processing chunk {chunk.chunk_idx + 1}/{len(chunk_plan)} (Frames {chunk.start_frame}..{chunk.end_frame - 1})...",
                    model="Meta SAM 2.1 Video Predictor",
                    action=f"Decoding bounded frame buffer [{chunk.start_frame}:{chunk.end_frame}]",
                    detail=f"Chunk {chunk.chunk_idx + 1} of {len(chunk_plan)} ({chunk.total_frames} frames)",
                )

                t_c_start = time.perf_counter()
                min_m_frame = chunk.commit_start if chunk.chunk_idx > 0 else 0
                chunk_frames, proxy_meta = frame_buffer.fetch_chunk(
                    start_frame=chunk.start_frame,
                    end_frame=chunk.end_frame,
                    keyframe_selector=selector,
                    min_motion_frame=min_m_frame,
                )
                t_chunk_prep_sec += (time.perf_counter() - t_c_start)
                scale_x = proxy_meta.get("scale_x", 1.0)
                scale_y = proxy_meta.get("scale_y", 1.0)
                if "chunk_keyframes" in proxy_meta:
                    keyframe_indices.update(proxy_meta["chunk_keyframes"])
                if "chunk_motion_metas" in proxy_meta:
                    motion_metas.update(proxy_meta["chunk_motion_metas"])

                sam2_tracker.init_video(
                    chunk_frames,
                    frame_offset=chunk.start_frame,
                    orig_hw=(proxy_meta.get("proxy_h", 480), proxy_meta.get("proxy_w", 640)),
                )

                # Carry forward active tracks into this chunk's SAM inference state
                # seeded from the exact overlap start frame (Fixes P1 Issue 1)
                if chunk.chunk_idx > 0:
                    seed_tracks = saved_overlap_start_tracks.get(chunk.start_frame, [])
                    for s_trk in seed_tracks:
                        m_reg = s_trk.get("mask")
                        if m_reg is not None:
                            try:
                                sam2_tracker.register_mask(
                                    frame_idx=chunk.start_frame,
                                    obj_id=s_trk["track_id"],
                                    binary_mask=m_reg,
                                    is_correction=False,
                                    metadata=s_trk["metadata"],
                                )
                            except Exception as reg_err:
                                _set_stage(log_entry=f"Warning registering coral {s_trk['track_id']} into chunk {chunk.chunk_idx}: {reg_err}")

                c_keyframes = [kf for kf in keyframe_indices if chunk.start_frame <= kf < chunk.end_frame]
                if not c_keyframes and chunk.chunk_idx == 0:
                    c_keyframes = [0]
                elif not c_keyframes and chunk.start_frame not in keyframe_indices:
                    c_keyframes = [chunk.start_frame]
                c_keyframes = sorted(list(set(c_keyframes)))

                incoming_chunk_overlap_tracks: Dict[int, List[Dict[str, Any]]] = {}
                tail_overlap_tracks: Dict[int, List[Dict[str, Any]]] = {}
                next_chunk_start = chunk_plan[chunk.chunk_idx + 1].start_frame if (chunk.chunk_idx + 1 < len(chunk_plan)) else None

                def _propagate_range(start_f: int, end_f: int):
                    """Propagates SAM masks over [start_f, end_f), updating tracks, overlap snapshots and committed frames."""
                    for out_f, det_tracks in sam2_tracker.propagate(
                        start_frame_idx=start_f,
                        end_frame_idx=end_f,
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

                        active = tracker.get_active_tracks()
                        _save_overlap_snapshot(out_f, active)
                        if chunk.has_overlap and chunk.reconcile_start <= out_f < chunk.reconcile_end:
                            incoming_chunk_overlap_tracks[out_f] = [t.to_dict() for t in active]
                            if out_f == chunk.reconcile_end - 1 and previous_chunk_overlap_frames:
                                rec_map = reconcile_chunk_identities(
                                    previous_chunk_overlap_frames,
                                    incoming_chunk_overlap_tracks,
                                    iou_threshold=0.50,
                                )
                                tracker.apply_id_reconciliation(rec_map)

                        if chunk.commit_start <= out_f < chunk.commit_end:
                            _record_frame_state(out_f, active)

                        if next_chunk_start is not None and out_f >= next_chunk_start:
                            tail_overlap_tracks[out_f] = [t.to_dict() for t in active]

                # If first keyframe is strictly after chunk.start_frame, propagate intervening frames
                if c_keyframes and c_keyframes[0] > chunk.start_frame:
                    _propagate_range(chunk.start_frame + 1, c_keyframes[0])

                # Process keyframes in this chunk
                for k_i, kf_frame_idx in enumerate(c_keyframes):
                    if job["cancel_requested"]:
                        break

                    # Step 1: Keyframe discovery & correction (reuse in-memory proxy & raw frames)
                    rel_f = kf_frame_idx - chunk.start_frame
                    kf_proc_rgb = chunk_frames[rel_f] if (0 <= rel_f < len(chunk_frames)) else None
                    kf_rgb = frame_buffer.get_keyframe_raw_rgb(kf_frame_idx)
                    if kf_proc_rgb is None and kf_rgb is not None:
                        kf_proc_rgb = cv2.resize(kf_rgb, (kf_proxy_w, kf_proxy_h), interpolation=cv2.INTER_AREA) if is_kf_proxy else kf_rgb

                    if kf_proc_rgb is not None:
                        points_per_side = opts.get("points_per_side", 16)
                        _set_stage(
                            stage=f"Keyframe at Frame {kf_frame_idx}: Segmenting benthic colonies...",
                            model="CoralSCOP SAM ViT-B",
                            action=f"Prompt grid segmentation ({points_per_side}x{points_per_side} points)",
                            detail=f"Frame {kf_frame_idx} ({kf_proc_rgb.shape[1]}x{kf_proc_rgb.shape[0]})",
                        )
                        t_disc_start = time.perf_counter()
                        masks_info, _ = run_segmentation(
                            model=sam_model,
                            image=kf_proc_rgb,
                            points_per_side=points_per_side,
                            pred_iou_thresh=opts.get("pred_iou_thresh", 0.50),
                            stability_score_thresh=opts.get("stability_thresh", 0.50),
                            min_mask_region_area=25,
                        )
                        t_discovery_sec += (time.perf_counter() - t_disc_start)

                        # Scale from the array actually segmented. It is usually the SAM2 chunk proxy
                        # (max_dim 480), not the 480p keyframe proxy, so a fixed ratio mis-scales it.
                        seg_h, seg_w = kf_proc_rgb.shape[:2]
                        if (seg_w, seg_h) != (width, height):
                            rescale_detections_to_full_res(masks_info, width / float(seg_w), height / float(seg_h))

                        tracker.match_keyframe_detections(
                            new_detections=masks_info,
                            frame_idx=kf_frame_idx,
                            frame_w=width,
                            frame_h=height,
                        )

                        # Register matched corrections into chunk SAM (Fixes P1 Issue 6)
                        matched_corrections = tracker.pop_matched_corrections()
                        for trk, det in matched_corrections:
                            m_corr = det.get("mask")
                            if m_corr is None and det.get("polygon"):
                                m_corr = np.zeros((height, width), dtype=np.uint8)
                                pts = np.array(det["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(m_corr, [pts], 1)
                                m_corr = m_corr.astype(bool)
                            if m_corr is not None:
                                try:
                                    sam2_tracker.register_mask(
                                        frame_idx=kf_frame_idx,
                                        obj_id=trk.track_id,
                                        binary_mask=m_corr,
                                        is_correction=True,
                                        metadata=track_metadata(trk),
                                    )
                                except Exception as corr_err:
                                    _set_stage(log_entry=f"Warning registering correction for coral {trk.track_id}: {corr_err}")

                        # Register new masks immediately into SAM; enrich taxonomy/bleaching asynchronously
                        new_tracks = tracker.get_tracks_needing_enrichment()
                        if new_tracks:
                            _set_stage(
                                stage=f"Keyframe at Frame {kf_frame_idx}: Registering & classifying {len(new_tracks)} new corals...",
                                model="BioCLIP ViT & YOLO11n Bleaching",
                                action="Registering masks immediately & scheduling async taxonomy inference",
                                detail=f"Extracting {len(new_tracks)} crops",
                            )

                            for trk in new_tracks:
                                m_reg = getattr(trk, "mask", None)
                                if m_reg is None and trk.polygon:
                                    m_poly = np.zeros((height, width), dtype=np.uint8)
                                    pts = np.array(trk.polygon, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.fillPoly(m_poly, [pts], 1)
                                    m_reg = m_poly.astype(bool)
                                if m_reg is not None:
                                    try:
                                        sam2_tracker.register_mask(
                                            frame_idx=kf_frame_idx,
                                            obj_id=trk.track_id,
                                            binary_mask=m_reg,
                                            is_correction=False,
                                                metadata=track_metadata(trk, kf_frame_idx),
                                        )
                                    except Exception as reg_err:
                                        _set_stage(log_entry=f"Warning registering coral {trk.track_id} into SAM: {reg_err}")

                            # Read full frame for crops if not loaded
                            if kf_rgb is None:
                                kf_rgb = frame_buffer.get_keyframe_raw_rgb(kf_frame_idx)

                            pil_img = Image.fromarray(kf_rgb)
                            task_tracks = list(new_tracks)
                            crops = crop_track_regions(pil_img, task_tracks, width, height)

                            if crops:
                                def _enrich_worker(c_crops, c_tracks, f_idx):
                                    t_c0 = time.perf_counter()
                                    t_res = classify_coral_taxonomy(c_crops, bioclip_bundle=bioclip_bundle, top_k=3)
                                    b_res = detect_coral_bleaching(c_crops, bleaching_model=bleaching_model)
                                    c_dur = time.perf_counter() - t_c0
                                    results = {}
                                    for k_idx, trk_item in enumerate(c_tracks):
                                        results[trk_item.track_id] = {
                                            "track_id": trk_item.track_id,
                                            "taxonomy": t_res[k_idx],
                                            "bleaching": b_res[k_idx],
                                            "first_frame": f_idx,
                                            "color_rgb": trk_item.color_rgb,
                                            "color_hex": trk_item.color_hex,
                                            "predicted_iou": trk_item.predicted_iou,
                                        }
                                    return {"duration_sec": c_dur, "results": results}

                                # Bound outstanding classification tasks (backpressure)
                                MAX_PENDING_ENRICHMENT = 4
                                while len(pending_enrichment_futures) >= MAX_PENDING_ENRICHMENT:
                                    _reap_enrichment_results(block=True)

                                fut = enrichment_executor.submit(_enrich_worker, crops, task_tracks, kf_frame_idx)
                                pending_enrichment_futures.append(fut)

                        # Retire absent colonies (Fixes P1 Issue 5)
                        retired_ids = tracker.pop_retired_ids()
                        if retired_ids:
                            sam2_tracker.retire_objects(retired_ids)

                    # Record keyframe if in commit range
                    active_tracks = tracker.get_active_tracks()
                    _save_overlap_snapshot(kf_frame_idx, active_tracks)
                    if chunk.has_overlap and chunk.reconcile_start <= kf_frame_idx < chunk.reconcile_end:
                        incoming_chunk_overlap_tracks[kf_frame_idx] = [t.to_dict() for t in active_tracks]
                        if kf_frame_idx == chunk.reconcile_end - 1 and previous_chunk_overlap_frames:
                            rec_map = reconcile_chunk_identities(
                                previous_chunk_overlap_frames,
                                incoming_chunk_overlap_tracks,
                                iou_threshold=0.50,
                            )
                            tracker.apply_id_reconciliation(rec_map)

                    if chunk.commit_start <= kf_frame_idx < chunk.commit_end:
                        _record_frame_state(kf_frame_idx, active_tracks)

                    if next_chunk_start is not None and kf_frame_idx >= next_chunk_start:
                        tail_overlap_tracks[kf_frame_idx] = [t.to_dict() for t in active_tracks]

                    # Propagate to next keyframe or chunk end
                    next_kf_or_end = c_keyframes[k_i + 1] if k_i + 1 < len(c_keyframes) else chunk.end_frame
                    if next_kf_or_end > kf_frame_idx + 1:
                        _propagate_range(kf_frame_idx + 1, next_kf_or_end)

                # Accumulate timings and complete chunk
                timings = sam2_tracker.get_timings()
                accumulated_prop_sec += timings.get("propagation_sec", 0.0)
                accumulated_prop_frames += timings.get("frames_propagated", 0)
                accumulated_transfer_sec += timings.get("transfer_sec", 0.0)
                accumulated_geometry_sec += timings.get("geometry_sec", 0.0)
                accumulated_init_sec += timings.get("initialization_sec", 0.0)
                accumulated_queue_wait_sec += timings.get("queue_wait_sec", 0.0)
                sam2_tracker.reset_timings()
                previous_chunk_overlap_frames = tail_overlap_tracks
                self.store.update_chunk_status(job_id, chunk.chunk_idx, "completed")

            # Flush any remaining batched frame writes
            _flush_pending_frames()

            # Await all background classification tasks and apply results through coordinator
            _reap_enrichment_results(block=True)

            if job["cancel_requested"]:
                # Covers cancels that arrive during the last chunk, when the chunk loop ends without seeing the flag
                _finish_cancelled()
            else:
                # Strict frame coverage verification (Fixes P1 Issue 4)
                missing_frames = set(range(total_frames)) - committed_frames
                if missing_frames:
                    sorted_missing = sorted(list(missing_frames))
                    raise RuntimeError(
                        f"Pipeline failed to achieve complete frame coverage: "
                        f"{len(missing_frames)} frames were not processed (missing range: {sorted_missing[0]}..{sorted_missing[-1]}). "
                        f"Committed {len(committed_frames)}/{total_frames} frames."
                    )

                with self._lock:
                    job["state"] = VideoJobState.FINALIZING
                self.store.set_job_state(job_id, VideoJobState.FINALIZING)

                _set_stage(
                    stage="Finalizing survey deliverables...",
                    model="Transect Exporter",
                    action="Generating analytics CSV and COCO dataset",
                    detail="Writing survey deliverables to cache",
                )

                time_series_data = self.store.get_time_series(job_id)
                frame_adapter = StoreFrameMapAdapter(self.store, job_id)
                export_mp4_enabled = opts.get("export_mp4", True)

                exports = run_deliverable_exports(
                    cache_dir=self.cache_dir,
                    job_id=job_id,
                    filename=job["filename"],
                    video_path=video_path,
                    width=width,
                    height=height,
                    total_frames=total_frames,
                    fps=video_fps,
                    tracks=list(tracker._tracks.values()),
                    time_series_data=time_series_data,
                    frame_adapter=frame_adapter,
                    export_mp4=export_mp4_enabled,
                    set_stage=_set_stage,
                )
                export_paths, export_errors, export_status = exports.paths, exports.errors, exports.status
                t_export_deliv_sec = exports.deliverables_sec
                t_export_mp4_sec = exports.mp4_sec

                elapsed_total = round(time.time() - pipeline_start_t, 2)
                _set_stage(
                    stage="Transect analysis completed",
                    model="Completed",
                    action="Ready for playback & review",
                    detail=f"Tracked {tracker.total_unique_colonies} unique corals in {elapsed_total}s ({job['analysis_fps']} FPS)",
                    log_entry=f"Survey finished: {tracker.total_unique_colonies} corals, {job['bleaching_prevalence_pct']}% bleaching across {len(committed_frames)} frames in {elapsed_total}s (export_status={export_status})",
                )

                tracker_timings = sam2_tracker.get_timings()
                total_prop_sec = accumulated_prop_sec + tracker_timings.get("propagation_sec", 0.0)
                total_transfer_sec = accumulated_transfer_sec + tracker_timings.get("transfer_sec", 0.0)
                total_geometry_sec = accumulated_geometry_sec + tracker_timings.get("geometry_sec", 0.0)
                total_queue_wait_sec = accumulated_queue_wait_sec + tracker_timings.get("queue_wait_sec", 0.0)
                pipeline_timings = build_pipeline_timings(
                    elapsed_total=elapsed_total,
                    compile_warmup_sec=compile_warmup_sec,
                    initialization_sec=accumulated_init_sec,
                    motion_scan_sec=t_motion_scan_sec,
                    chunk_prep_sec=t_chunk_prep_sec,
                    propagation_sec=total_prop_sec,
                    transfer_sec=total_transfer_sec,
                    geometry_sec=total_geometry_sec,
                    queue_wait_sec=total_queue_wait_sec,
                    discovery_sec=t_discovery_sec,
                    classification_sec=t_classification_sec,
                    persistence_sec=t_persistence_sec,
                    export_deliverables_sec=t_export_deliv_sec,
                    export_mp4_sec=t_export_mp4_sec,
                    unique_colonies=tracker.total_unique_colonies,
                    keyframes_count=len(keyframe_indices),
                    committed_frames=len(committed_frames),
                )

                with self._lock:
                    job["timings"] = pipeline_timings
                    job["state"] = VideoJobState.COMPLETED
                    job["completed_at"] = time.time()
                    job["export_status"] = export_status
                    job["export_paths"] = export_paths
                    job["export_errors"] = export_errors

                self.store.set_job_state(
                    job_id=job_id,
                    state=VideoJobState.COMPLETED,
                    completed_at=time.time(),
                    export_paths=export_paths,
                    export_status=export_status,
                    timings=pipeline_timings,
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
            try:
                if frame_buffer is not None:
                    frame_buffer.close()
            except Exception:
                pass
            try:
                prefetch_executor.shutdown(wait=False)
            except Exception:
                pass
            try:
                enrichment_executor.shutdown(wait=False)
            except Exception:
                pass
            if sam2_tracker:
                sam2_tracker.reset()
