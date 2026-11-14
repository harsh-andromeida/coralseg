"""
Asynchronous Video Inference Engine and Transect Processor for ReefVision.
Colonies are tracked with camera-motion-driven optical flow (see core/video/flow_tracking.py): CoralSCOP segments
ego-motion keyframes in the background, BioCLIP/YOLO classify each new colony once, and every frame, track and
survey metric is persisted durably in SQLite.
"""

import os
import time
import queue
import threading
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from core.models import (
    acquire_active_job,
    release_active_job,
    reserve_memory_for_auxiliary_models,
)
from core.video.preprocessor import standardize_video_to_25fps
from core.video.flow_tracking import FlowTracker
from core.video.tracker import CoralTrack
from core.video.store import VideoJobStore
from core.video.pipeline_stages import (
    build_pipeline_timings,
    colony_color,
    labels_to_geometry,
    run_deliverable_exports,
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
    snapshot configuration, durable SQLite storage, and isolated error reporting.
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
        self._recover_orphaned_jobs()

        # Bounded job queue: default 1 active video job per accelerator
        self._job_queue = queue.Queue()
        self._queue_worker_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
        self._queue_worker_thread.start()

    def close(self):
        """Cleanly releases storage."""
        self.store.close()

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
        """Returns the job that is currently live (queued, running or paused), else None.

        Finished, failed or cancelled jobs are never returned, so reopening the app does not
        resurrect old results from the persisted job history.
        """
        live_states = (
            VideoJobState.QUEUED,
            VideoJobState.PREPARING,
            VideoJobState.PROCESSING,
            VideoJobState.FINALIZING,
            VideoJobState.PAUSED,
            VideoJobState.CANCELLING,
        )
        with self._lock:
            live = [j for j in self._jobs.values() if j.get("state") in live_states]
        if not live:
            return None
        newest = max(live, key=lambda j: j.get("created_at", 0))
        return self.get_job(newest["job_id"])

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
            "overlap_threshold": 0.70,
            "min_stride": 10,
            "max_stride": 25,
            "points_per_side": 16,
            "pred_iou_thresh": 0.50,
            "stability_thresh": 0.50,
            "device_pref": "auto",
            "classify": True,
            "export_mp4": True,
        }
        if params:
            opts.update({k: v for k, v in params.items() if k in opts})

        if opts["min_stride"] > opts["max_stride"]:
            raise ValueError(f"min_stride ({opts['min_stride']}) cannot exceed max_stride ({opts['max_stride']})")

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
            "current_model": "Optical-flow tracker + CoralSCOP",
            "current_action": "Waiting in processing queue",
            "stage_detail": f"Video: {probe['total_frames']} frames @ {probe['width']}x{probe['height']}",
            "stage_progress_pct": 0.0,
            "logs": [
                f"[{time.strftime('%H:%M:%S')}] Enqueued transect job on '{probe['filename']}' ({probe['width']}x{probe['height']}, {probe['total_frames']} frames @ 25fps)",
                f"[{time.strftime('%H:%M:%S')}] Keyframe stride {opts['min_stride']}-{opts['max_stride']}, overlap gate {opts['overlap_threshold']}",
            ],
            "tracker_mode": "flow",
        }

        # Persist to durable SQLite database
        self.store.create_job(
            job_id=job_id,
            video_path=active_video_path,
            raw_video_path=video_path,
            filename=probe["filename"],
            metadata=probe,
            effective_options=opts,
            tracker_mode="flow",
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
        """Worker loop: decode, flow-track, persist every frame, classify new colonies, export deliverables."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return

        with self._lock:
            job["state"] = VideoJobState.PREPARING
            job["started_at"] = time.time()
        self.store.set_job_state(job_id, VideoJobState.PREPARING, started_at=time.time())

        cap = cv2.VideoCapture(video_path)

        def _fail(stage: str, message: str):
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = message
                job["failed_stage"] = stage
            self.store.set_job_state(job_id, VideoJobState.ERROR, error=message, failed_stage=stage)

        if not cap.isOpened():
            _fail("video_demux", f"Failed to open video file: {video_path}")
            return

        total_frames = job["metadata"]["total_frames"]
        width = job["metadata"]["width"]
        height = job["metadata"]["height"]
        video_fps = job["metadata"]["fps"]
        device_pref = opts.get("device_pref", "auto")

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
                    job.setdefault("logs", []).append(f"[{time.strftime('%H:%M:%S')}] {log_entry}")
                    if len(job["logs"]) > 100:
                        job["logs"] = job["logs"][-100:]
                    print(f"  ⚡ [Pipeline] {log_entry}", flush=True)

        def _finish_cancelled():
            with self._lock:
                job["state"] = VideoJobState.CANCELLED
            self.store.set_job_state(job_id, VideoJobState.CANCELLED)
            _set_stage(stage="Analysis cancelled", model="Cancelled", action="Stopped by user",
                       log_entry="Video analysis cancelled by user")

        # 1. Memory admission and foundation models
        t_model_load = time.perf_counter()
        try:
            ok, mem_msg = reserve_memory_for_auxiliary_models(device_pref)
            if not ok:
                _set_stage(log_entry=f"Memory admission warning: {mem_msg}")
            _set_stage(
                stage="Loading foundation models...",
                model="CoralSCOP, BioCLIP, YOLO11n",
                action="Initializing inference sessions",
                log_entry="Loading foundation models into resident memory",
            )
            sam_model = self.sam_loader()
            classify = None
            if opts.get("classify", True):
                classify = {"bioclip": self.bioclip_loader(), "bleaching": self.bleaching_loader()}
            provider = sam_model["encoder"].get_providers()[0]
            _set_stage(stage="Foundation models loaded", detail=f"Inference sessions active ({provider})",
                       log_entry=f"Foundation models active ({provider})")
        except Exception as e:
            print(f"  ❌ [Pipeline Error] Model loading failed: {e}", flush=True)
            traceback.print_exc()
            _fail("model_load", f"Model loading failed: {e}")
            cap.release()
            return
        t_model_load = time.perf_counter() - t_model_load

        if job["cancel_requested"]:
            _finish_cancelled()
            cap.release()
            return

        tracker = FlowTracker(
            sam_model, width, height, classify=classify,
            overlap_threshold=opts["overlap_threshold"], min_stride=opts["min_stride"], max_stride=opts["max_stride"],
            points_per_side=opts["points_per_side"], pred_iou_thresh=opts["pred_iou_thresh"],
            stability_thresh=opts["stability_thresh"],
        )
        with self._lock:
            job["state"] = VideoJobState.PROCESSING
        self.store.set_job_state(job_id, VideoJobState.PROCESSING)
        _set_stage(
            stage="Tracking colonies...",
            model="Optical-flow tracker + CoralSCOP",
            action="Warping colony masks with dense optical flow; segmenting keyframes in the background",
            detail=f"{total_frames} frames",
        )

        tracks: Dict[int, CoralTrack] = {}
        keyframe_indices = {0}
        committed_frames = set()
        pending_frame_writes: List[Dict[str, Any]] = []
        t_decode = t_geometry = t_persistence = 0.0
        t_export_deliv = t_export_mp4 = 0.0
        last_progress_t = 0.0
        pipeline_start = time.time()

        def _apply_classifications():
            for cid, res in tracker.drain_classifications():
                trk = tracks.get(cid)
                if trk is None:
                    continue
                trk.taxonomy, trk.bleaching, trk.needs_enrichment = res["taxonomy"], res["bleaching"], False
                self.store.save_track(
                    job_id=job_id, track_id=cid, global_colony_id=cid,
                    first_frame=trk.first_seen_frame, last_frame=trk.first_seen_frame,
                    genus=trk.taxonomy.get("genus", "Coral"), growth_form=trk.taxonomy.get("growth_form", "-"),
                    taxon_conf=float(trk.taxonomy.get("confidence_pct", 85.0)),
                    condition=str(trk.bleaching.get("status", "Healthy")),
                    condition_conf=float(trk.bleaching.get("confidence_pct", 90.0)),
                    color_rgb=trk.color_rgb, metadata={"color_hex": trk.color_hex, "predicted_iou": trk.predicted_iou},
                )
                self.store.finalize_track_classification(
                    job_id=job_id, track_id=cid, taxonomy=trk.taxonomy, bleaching=trk.bleaching,
                )

        def _flush_frames():
            nonlocal t_persistence
            if pending_frame_writes:
                t0 = time.perf_counter()
                self.store.save_frames_batch(job_id=job_id, frames_data=pending_frame_writes)
                t_persistence += time.perf_counter() - t0
                pending_frame_writes.clear()

        def _record_frame(cur_f: int, active: List[CoralTrack], step):
            committed_frames.add(cur_f)
            bleached = sum(1 for t in active if t.bleaching.get("is_bleached", False))
            bleach_pct = bleached / max(len(active), 1) * 100.0
            genus_counts: Dict[str, int] = {}
            for t in active:
                g = t.taxonomy.get("genus", "Coral")
                genus_counts[g] = genus_counts.get(g, 0) + 1
            record = {
                "frame_index": cur_f,
                "timestamp_sec": round(cur_f / max(video_fps, 1.0), 2),
                "coral_coverage_pct": round(sum(t.area_pct for t in active), 2),
                "active_corals_count": len(active),
                "cumulative_unique_corals": len(tracks),
                "bleached_corals_count": bleached,
                "healthy_corals_count": len(active) - bleached,
                "bleaching_prevalence_pct": round(bleach_pct, 1),
                "dominant_genus": max(genus_counts, key=genus_counts.get) if genus_counts else "Coral",
                "is_keyframe": cur_f in keyframe_indices,
                "camera_motion_dx": step.motion_dx,
                "camera_motion_dy": step.motion_dy,
            }
            pending_frame_writes.append({
                "frame_idx": cur_f, "timestamp_sec": record["timestamp_sec"], "is_keyframe": record["is_keyframe"],
                "tracks": [t.to_dict() for t in active], "metrics": record,
            })
            if len(pending_frame_writes) >= 32 or record["is_keyframe"] or cur_f == total_frames - 1:
                _flush_frames()

            elapsed = max(0.001, time.time() - pipeline_start)
            fps_now = round(len(committed_frames) / elapsed, 1)
            nonlocal last_progress_t
            now = time.time()
            if now - last_progress_t >= 0.25 or record["is_keyframe"] or cur_f == total_frames - 1:
                self.store.update_job_progress(
                    job_id=job_id, current_frame=cur_f + 1,
                    progress_pct=round(len(committed_frames) / max(total_frames, 1) * 100.0, 1),
                    fps_realtime=fps_now, tracker_fps=fps_now, analysis_fps=fps_now, elapsed_sec=round(elapsed, 2),
                    active_corals_count=len(active), total_unique_corals=len(tracks),
                    bleaching_prevalence_pct=round(bleach_pct, 1), keyframes_count=len(keyframe_indices),
                    keyframe_indices=sorted(keyframe_indices),
                )
                last_progress_t = now
            with self._lock:
                job.update(
                    current_frame=cur_f + 1,
                    progress_pct=round(len(committed_frames) / max(total_frames, 1) * 100.0, 1),
                    active_corals_count=len(active), total_unique_corals=len(tracks),
                    bleaching_prevalence_pct=round(bleach_pct, 1), analysis_fps=fps_now, tracker_fps=fps_now,
                    fps_realtime=fps_now, elapsed_sec=round(elapsed, 2),
                    keyframes_count=len(keyframe_indices), keyframe_indices=sorted(keyframe_indices),
                )

        try:
            frame_idx = 0
            while True:
                if job["cancel_requested"]:
                    break
                while job["pause_requested"] and not job["cancel_requested"]:
                    time.sleep(0.1)

                t0 = time.perf_counter()
                ok, frame_bgr = cap.read()
                t_decode += time.perf_counter() - t0
                if not ok:
                    break

                step = tracker.step(frame_bgr)
                if step.is_keyframe or frame_idx == 0:
                    keyframe_indices.add(frame_idx)

                t0 = time.perf_counter()
                geometry = labels_to_geometry(step.labels, width, height)
                for cid, g in geometry.items():
                    trk = tracks.get(cid)
                    if trk is None:
                        tracks[cid] = CoralTrack(
                            track_id=cid, bbox=g["bbox"], polygon=g["polygon"], area_px=g["area_px"],
                            area_pct=g["area_pct"], predicted_iou=0.85, centroid=g["centroid"],
                            color_rgb=colony_color(cid), frame_idx=frame_idx,
                        )
                    else:
                        trk.update_geometry(
                            bbox=g["bbox"], polygon=g["polygon"], centroid=g["centroid"], area_px=g["area_px"],
                            area_pct=g["area_pct"], frame_idx=frame_idx,
                        )
                for cid, trk in tracks.items():
                    trk.active = cid in geometry
                t_geometry += time.perf_counter() - t0

                _apply_classifications()
                _record_frame(frame_idx, [tracks[c] for c in geometry], step)
                frame_idx += 1

            tracker.finish()
            _apply_classifications()
            _flush_frames()

            if job["cancel_requested"]:
                _finish_cancelled()
            else:
                missing = set(range(total_frames)) - committed_frames
                if missing:
                    ordered = sorted(missing)
                    raise RuntimeError(
                        f"Pipeline failed to achieve complete frame coverage: {len(missing)} frames were not "
                        f"processed (missing range: {ordered[0]}..{ordered[-1]}). "
                        f"Committed {len(committed_frames)}/{total_frames} frames."
                    )

                with self._lock:
                    job["state"] = VideoJobState.FINALIZING
                self.store.set_job_state(job_id, VideoJobState.FINALIZING)
                _set_stage(stage="Finalizing survey deliverables...", model="Transect Exporter",
                           action="Generating analytics CSV and COCO dataset", detail="Writing survey deliverables to cache")

                exports = run_deliverable_exports(
                    cache_dir=self.cache_dir, job_id=job_id, filename=job["filename"], video_path=video_path,
                    width=width, height=height, total_frames=total_frames, fps=video_fps,
                    tracks=list(tracks.values()), time_series_data=self.store.get_time_series(job_id),
                    frame_adapter=StoreFrameMapAdapter(self.store, job_id),
                    export_mp4=opts.get("export_mp4", True), set_stage=_set_stage,
                )
                t_export_deliv, t_export_mp4 = exports.deliverables_sec, exports.mp4_sec

                elapsed_total = round(time.time() - pipeline_start, 2)
                _set_stage(
                    stage="Transect analysis completed", model="Completed", action="Ready for playback & review",
                    detail=f"Tracked {len(tracks)} unique corals in {elapsed_total}s ({job['analysis_fps']} FPS)",
                    log_entry=(f"Survey finished: {len(tracks)} corals, {job['bleaching_prevalence_pct']}% bleaching "
                               f"across {len(committed_frames)} frames in {elapsed_total}s (export_status={exports.status})"),
                )
                tm = tracker.timings
                pipeline_timings = build_pipeline_timings(
                    elapsed_total=elapsed_total, model_load_sec=t_model_load, decode_sec=t_decode + tm["proxy"],
                    ego_motion_sec=tm["ego_motion"], flow_sec=tm["flow"] + tm["warp"], geometry_sec=t_geometry,
                    keyframe_blocked_sec=tm["blocked_on_keyframe"], keyframe_segmentation_sec=tm["keyframe_segmentation"],
                    classification_sec=tm["classification"], persistence_sec=t_persistence,
                    export_deliverables_sec=t_export_deliv, export_mp4_sec=t_export_mp4,
                    unique_colonies=len(tracks), keyframes_count=len(keyframe_indices),
                    committed_frames=len(committed_frames),
                )
                with self._lock:
                    job["timings"] = pipeline_timings
                    job["state"] = VideoJobState.COMPLETED
                    job["completed_at"] = time.time()
                    job["export_status"] = exports.status
                    job["export_paths"] = exports.paths
                    job["export_errors"] = exports.errors
                self.store.set_job_state(
                    job_id=job_id, state=VideoJobState.COMPLETED, completed_at=time.time(),
                    export_paths=exports.paths, export_status=exports.status, timings=pipeline_timings,
                )

        except Exception as e:
            traceback.print_exc()
            err_msg = f"Pipeline execution error: {str(e)}"
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = err_msg
            self.store.set_job_state(
                job_id=job_id, state=VideoJobState.ERROR, error=err_msg,
                failed_stage=job.get("failed_stage"), failed_frame_range=str(job.get("failed_frame_range")),
            )
            _set_stage(stage="Pipeline execution halted on error", model="Error", detail=err_msg,
                       log_entry=f"ERROR: {err_msg}")
        finally:
            cap.release()
            tracker.cancel()
