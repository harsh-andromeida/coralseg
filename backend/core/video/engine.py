"""
Asynchronous Video Inference Engine and Transect Processor for ReefVision.
Coordinates frame ingestion, ego-motion keyframe selection, benthic tracking,
on-demand BioCLIP/YOLO enrichment, and real-time status streaming.
"""

import os
import time
import shutil
import threading
from typing import Dict, Any, List, Optional, Callable
from PIL import Image
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from core.video.preprocessor import standardize_video_to_25fps, extract_video_frames_to_disk
from core.video.keyframe_selector import EgoMotionKeyframeSelector
from core.video.tracker import BenthicColonyTracker, CoralTrack
from core.video.sam2_tracker import SAM2BenthicTracker
from core.video.export import render_video_overlay, export_annotated_video, export_transect_csv, export_coco_video
from core.segmentation import run_segmentation
from core.taxonomy import classify_coral_taxonomy, detect_coral_bleaching


class VideoJobState:
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class VideoProcessor:
    """
    Manages asynchronous underwater video processing pipelines with live status and playback caching.
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

        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def probe_video(self, video_path: str) -> Dict[str, Any]:
        """Probes video metadata using OpenCV."""
        if not _HAS_CV2:
            raise RuntimeError("OpenCV is required to process video.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
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
        """Initializes and launches a background video processing job with 25 FPS standardization."""
        job_id = f"vjob_{int(time.time() * 1000)}"

        # 1. Standardize video to constant 25.0 FPS grid
        storage_dir = os.path.dirname(os.path.abspath(video_path))
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        std_25fps_path = os.path.join(storage_dir, f"{base_name}_25fps.mp4")

        try:
            if not os.path.isfile(std_25fps_path) or os.path.abspath(video_path) != os.path.abspath(std_25fps_path):
                probe = standardize_video_to_25fps(video_path, std_25fps_path)
                active_video_path = std_25fps_path
            else:
                active_video_path = std_25fps_path
                probe = self.probe_video(std_25fps_path)
        except Exception as e:
            active_video_path = video_path
            probe = self.probe_video(video_path)

        opts = {
            "overlap_threshold": 0.75,
            "min_stride": 5,
            "max_stride": 30,
            "points_per_side": 12,
            "iou_match_threshold": 0.30,
            "pred_iou_thresh": 0.45,
            "stability_thresh": 0.45,
            "device_pref": "auto",
        }
        if params:
            opts.update(params)

        job_info = {
            "job_id": job_id,
            "video_path": active_video_path,
            "raw_video_path": video_path,
            "filename": probe["filename"],
            "metadata": probe,
            "state": VideoJobState.PROCESSING,
            "current_frame": 0,
            "total_frames": probe["total_frames"],
            "fps_realtime": 0.0,
            "progress_pct": 0.0,
            "active_corals_count": 0,
            "total_unique_corals": 0,
            "bleaching_prevalence_pct": 0.0,
            "keyframes_count": 0,
            "keyframe_indices": [],
            "error": None,
            "cancel_requested": False,
            "pause_requested": False,
            "time_series": [],
            "frame_tracks_map": {},
            "all_tracks": {},
            "created_at": time.time(),
            "completed_at": None,
            "export_paths": {},
        }

        with self._lock:
            self._jobs[job_id] = job_info

        worker_thread = threading.Thread(
            target=self._run_job_pipeline,
            args=(job_id, active_video_path, opts),
            daemon=True,
        )
        worker_thread.start()
        return job_id

    def pause_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job["state"] == VideoJobState.PROCESSING:
                job["pause_requested"] = True
                job["state"] = VideoJobState.PAUSED
                return True
        return False

    def resume_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job["state"] == VideoJobState.PAUSED:
                job["pause_requested"] = False
                job["state"] = VideoJobState.PROCESSING
                return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["cancel_requested"] = True
                job["state"] = VideoJobState.CANCELLED
                return True
        return False

    def get_frame_annotation(self, job_id: str, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Retrieves tracking results and rendered base64 for a specific frame."""
        job = self.get_job(job_id)
        if not job:
            return None

        tracks = job["frame_tracks_map"].get(frame_idx, [])
        return {
            "job_id": job_id,
            "frame_idx": frame_idx,
            "is_keyframe": frame_idx in job["keyframe_indices"],
            "tracks": [t.to_dict() for t in tracks],
            "active_count": len(tracks),
        }

    def _run_job_pipeline(self, job_id: str, video_path: str, opts: Dict[str, Any]):
        """Main background worker execution loop."""
        job = self.get_job(job_id)
        if not job:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = "Failed to open video file."
            return

        total_frames = job["metadata"]["total_frames"]
        width = job["metadata"]["width"]
        height = job["metadata"]["height"]
        video_fps = job["metadata"]["fps"]

        # 1. Load models
        try:
            sam_model = self.sam_loader()
            bioclip_bundle = self.bioclip_loader()
            bleaching_model = self.bleaching_loader()
        except Exception as e:
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = f"Model loading failed: {str(e)}"
            cap.release()
            return

        # 1. Pre-evaluate keyframes using high-speed optical flow (~1ms per frame)
        selector = EgoMotionKeyframeSelector(
            overlap_threshold=opts["overlap_threshold"],
            min_stride=opts["min_stride"],
            max_stride=opts["max_stride"],
        )
        tracker = BenthicColonyTracker(
            iou_match_threshold=opts["iou_match_threshold"],
        )

        detected_kfs = [0]
        motion_metas = {}
        scan_cap = cv2.VideoCapture(video_path)
        f_scan = 0
        while True:
            ret, f_bgr = scan_cap.read()
            if not ret or f_bgr is None:
                break
            f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            is_kf, m_meta = selector.evaluate_frame(f_rgb)
            motion_metas[f_scan] = m_meta
            if is_kf and f_scan > 0:
                detected_kfs.append(f_scan)
            f_scan += 1
        scan_cap.release()

        keyframe_indices = sorted(list(set(detected_kfs)))

        # 2. Initialize SAM 2 Video Predictor
        storage_dir = os.path.dirname(os.path.abspath(video_path))
        frames_dir = os.path.join(storage_dir, f"frames_{job_id}")
        sam2_tracker = SAM2BenthicTracker(device=opts.get("device_pref", "auto"))
        has_sam2 = False

        try:
            extract_video_frames_to_disk(video_path, frames_dir)
            sam2_tracker.init_video(frames_dir)
            has_sam2 = True
        except Exception as sam2_err:
            has_sam2 = False

        with self._lock:
            job["keyframe_indices"] = list(keyframe_indices)
            job["keyframes_count"] = len(keyframe_indices)

        fps_clock = time.time()
        frames_processed_clock = 0

        try:
            # Process in windows between keyframes
            for w_idx in range(len(keyframe_indices)):
                if job["cancel_requested"]:
                    with self._lock:
                        job["state"] = VideoJobState.CANCELLED
                    break

                while job["pause_requested"] and not job["cancel_requested"]:
                    time.sleep(0.1)

                kf_frame_idx = keyframe_indices[w_idx]
                next_kf_idx = keyframe_indices[w_idx + 1] if w_idx + 1 < len(keyframe_indices) else total_frames
                window_length = next_kf_idx - kf_frame_idx

                # --- KEYFRAME STEP (Change > Threshold): Run SAM, BioCLIP, YOLO Bleaching ---
                cap.set(cv2.CAP_PROP_POS_FRAMES, kf_frame_idx)
                ret, kf_bgr = cap.read()
                if ret and kf_bgr is not None:
                    kf_rgb = cv2.cvtColor(kf_bgr, cv2.COLOR_BGR2RGB)
                    points_per_side = opts.get("points_per_side", 12)
                    masks_info, _ = run_segmentation(
                        model=sam_model,
                        image=kf_rgb,
                        points_per_side=points_per_side,
                        pred_iou_thresh=opts.get("pred_iou_thresh", 0.45),
                        stability_score_thresh=opts.get("stability_thresh", 0.45),
                        min_mask_region_area=80,
                    )

                    tracker.match_keyframe_detections(
                        new_detections=masks_info,
                        frame_idx=kf_frame_idx,
                        frame_w=width,
                        frame_h=height,
                    )

                    # Enrich newly spawned tracks
                    new_tracks = tracker.get_tracks_needing_enrichment()
                    if new_tracks:
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

                                # Register newly discovered coral into SAM 2 tracker
                                if has_sam2:
                                    m_reg = getattr(trk, "mask", None)
                                    if m_reg is None and trk.polygon:
                                        m_reg = np.zeros((height, width), dtype=np.uint8)
                                        pts = np.array(trk.polygon, dtype=np.int32).reshape((-1, 1, 2))
                                        cv2.fillPoly(m_reg, [pts], 1)
                                        m_reg = m_reg.astype(bool)
                                    if m_reg is not None:
                                        try:
                                            sam2_tracker.register_coral(
                                                frame_idx=kf_frame_idx,
                                                track_id=trk.track_id,
                                                binary_mask=m_reg,
                                                taxonomy=trk.taxonomy,
                                                bleaching=trk.bleaching,
                                                color_hex=trk.color_hex,
                                                predicted_iou=trk.predicted_iou,
                                            )
                                        except Exception:
                                            pass

                # --- CONTINUOUS TRACKING STEP (Propagate Across Intervening Frames) ---
                sam2_window_preds = {}
                if has_sam2 and sam2_tracker.total_tracks_count > 0:
                    try:
                        sam2_window_preds = sam2_tracker.propagate_window(
                            start_frame_idx=kf_frame_idx,
                            num_frames=window_length,
                        )
                    except Exception:
                        sam2_window_preds = {}

                # Record frame-by-frame tracks and update telemetry
                for cur_f in range(kf_frame_idx, next_kf_idx):
                    if job["cancel_requested"]:
                        break

                    if cur_f in sam2_window_preds:
                        # Update tracker with exact SAM 2 masks
                        for det in sam2_window_preds[cur_f]:
                            trk_id = det["track_id"]
                            if trk_id in tracker._tracks:
                                t = tracker._tracks[trk_id]
                                t.bbox = det["bbox"]
                                t.centroid = det["centroid"]
                                t.polygon = det["polygon"]
                                t.area_px = det["area_px"]
                                t.mask = det["mask"]
                                t.last_seen_frame = cur_f
                                t.time_since_update = 0
                                t.history.append((det["bbox"], det["polygon"]))
                    elif cur_f > kf_frame_idx:
                        # Fallback homography warping for non-keyframes
                        m_meta = motion_metas.get(cur_f, {})
                        step_H = m_meta.get("step_H", np.eye(3, dtype=np.float32))
                        tracker.warp_tracks_with_motion(
                            step_H=step_H,
                            frame_w=width,
                            frame_h=height,
                            frame_idx=cur_f,
                        )

                    active_tracks = tracker.get_active_tracks()
                    bleached_count = sum(1 for t in active_tracks if t.bleaching.get("is_bleached", False))
                    healthy_count = len(active_tracks) - bleached_count
                    bleach_pct = (bleached_count / max(len(active_tracks), 1)) * 100.0

                    genus_counts: Dict[str, int] = {}
                    for t in active_tracks:
                        g = t.taxonomy.get("genus", "Coral")
                        genus_counts[g] = genus_counts.get(g, 0) + 1
                    dom_genus = max(genus_counts, key=genus_counts.get) if genus_counts else "Coral"

                    timestamp_sec = round(cur_f / max(video_fps, 1.0), 2)
                    m_meta = motion_metas.get(cur_f, {})
                    record = {
                        "frame_index": cur_f,
                        "timestamp_sec": timestamp_sec,
                        "active_corals_count": len(active_tracks),
                        "cumulative_unique_corals": tracker._next_track_id - 1,
                        "bleached_corals_count": bleached_count,
                        "healthy_corals_count": healthy_count,
                        "bleaching_prevalence_pct": round(bleach_pct, 1),
                        "dominant_genus": dom_genus,
                        "is_keyframe": cur_f in keyframe_indices,
                        "camera_motion_dx": m_meta.get("motion_dx", 0.0),
                        "camera_motion_dy": m_meta.get("motion_dy", 0.0),
                    }

                    with self._lock:
                        job["current_frame"] = cur_f + 1
                        job["progress_pct"] = round(((cur_f + 1) / max(total_frames, 1)) * 100.0, 1)
                        job["active_corals_count"] = len(active_tracks)
                        job["total_unique_corals"] = tracker._next_track_id - 1
                        job["bleaching_prevalence_pct"] = round(bleach_pct, 1)
                        job["frame_tracks_map"][cur_f] = list(active_tracks)
                        job["time_series"].append(record)

                    frames_processed_clock += 1
                    now = time.time()
                    if now - fps_clock >= 0.5:
                        inst_fps = frames_processed_clock / max(now - fps_clock, 0.001)
                        with self._lock:
                            job["fps_realtime"] = round(inst_fps, 1)
                        fps_clock = now
                        frames_processed_clock = 0

            # Processing finished successfully
            if not job["cancel_requested"]:
                # Auto-generate export files in cache_dir
                annotated_mp4 = os.path.join(self.cache_dir, f"{job_id}_annotated.mp4")
                transect_csv = os.path.join(self.cache_dir, f"{job_id}_transect.csv")
                coco_json = os.path.join(self.cache_dir, f"{job_id}_coco_video.json")

                try:
                    export_annotated_video(
                        input_video_path=video_path,
                        output_mp4_path=annotated_mp4,
                        frame_tracks_map=job["frame_tracks_map"],
                    )
                    export_transect_csv(
                        output_csv_path=transect_csv,
                        time_series_records=job["time_series"],
                    )
                    export_coco_video(
                        output_json_path=coco_json,
                        video_filename=job["filename"],
                        width=width,
                        height=height,
                        total_frames=total_frames,
                        fps=video_fps,
                        all_tracks=list(tracker._tracks.values()),
                        frame_tracks_map=job["frame_tracks_map"],
                    )
                except Exception as export_err:
                    pass

                with self._lock:
                    job["state"] = VideoJobState.COMPLETED
                    job["completed_at"] = time.time()
                    job["export_paths"] = {
                        "annotated_mp4": annotated_mp4 if os.path.exists(annotated_mp4) else None,
                        "transect_csv": transect_csv if os.path.exists(transect_csv) else None,
                        "coco_json": coco_json if os.path.exists(coco_json) else None,
                    }

        except Exception as e:
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = f"Pipeline execution error: {str(e)}"
        finally:
            cap.release()
            if 'frames_dir' in locals() and os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir, ignore_errors=True)
