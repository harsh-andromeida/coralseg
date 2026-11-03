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

    def get_active_job(self) -> Optional[Dict[str, Any]]:
        """Retrieves currently running/paused job, or the most recent video job."""
        with self._lock:
            if not self._jobs:
                return None
            # Return currently processing or paused job first
            for j in reversed(list(self._jobs.values())):
                if j.get("state") in (VideoJobState.PROCESSING, VideoJobState.PAUSED):
                    return j
            # Fallback to the latest job
            sorted_jobs = sorted(self._jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)
            return sorted_jobs[0] if sorted_jobs else None

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
            "min_stride": 15,
            "max_stride": 45,
            "points_per_side": 16,
            "iou_match_threshold": 0.30,
            "pred_iou_thresh": 0.50,
            "stability_thresh": 0.50,
            "device_pref": "auto",
            "tracker_mode": "sam2",
        }
        if params:
            opts.update(params)

        tracker_mode = opts.get("tracker_mode", "optical_flow")

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
            "current_stage": "Initializing transect pipeline...",
            "current_model": "Video Demuxer / 25 FPS Grid",
            "current_action": "Preparing video stream and loading neural models",
            "stage_detail": f"Standardized 25.0 FPS video grid ({probe['total_frames']} frames @ {probe['width']}x{probe['height']})",
            "stage_progress_pct": 0.0,
            "logs": [
                f"[{time.strftime('%H:%M:%S')}] Started transect tracking on '{probe['filename']}' ({probe['width']}x{probe['height']}, {probe['total_frames']} frames @ 25fps)",
                f"[{time.strftime('%H:%M:%S')}] Engine configured: mode={tracker_mode}, min_stride={opts['min_stride']}, max_stride={opts['max_stride']}, points={opts['points_per_side']}x{opts['points_per_side']}",
            ],
            "tracker_mode": tracker_mode,
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

        frames_dir: Optional[str] = None
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
        tracker_mode = opts.get("tracker_mode", "optical_flow")

        def _set_stage(
            stage: str,
            model: Optional[str] = None,
            action: Optional[str] = None,
            detail: Optional[str] = None,
            log_entry: Optional[str] = None,
        ):
            with self._lock:
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
                    if len(job["logs"]) > 60:
                        job["logs"] = job["logs"][-60:]

        # 1. Load models
        try:
            _set_stage(
                stage="Loading foundation models into RAM...",
                model="CoralSCOP SAM, BioCLIP, YOLO11n",
                action="Initializing ONNX inference sessions",
                log_entry="Loading foundation models: CoralSCOP SAM ViT-B, BioCLIP ViT, and Bleaching YOLO11n",
            )
            sam_model = self.sam_loader()
            bioclip_bundle = self.bioclip_loader()
            bleaching_model = self.bleaching_loader()
            _set_stage(
                stage="Foundation models loaded",
                detail="Inference sessions active in system memory",
                log_entry="Foundation models loaded successfully into RAM",
            )
        except Exception as e:
            with self._lock:
                job["state"] = VideoJobState.ERROR
                job["error"] = f"Model loading failed: {str(e)}"
            cap.release()
            return

        # 2. Pre-evaluate keyframes using high-speed optical flow (~1-2ms per frame)
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

        _set_stage(
            stage="Analyzing camera ego-motion & seafloor overlap...",
            model="Lucas-Kanade Optical Flow",
            action="Computing camera translation & benthic scene overlap",
            detail=f"Scanning frame 0/{total_frames}",
        )

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

            if f_scan % 25 == 0 or f_scan == total_frames:
                _set_stage(
                    stage=f"Analyzing camera motion: {f_scan}/{total_frames} frames ({int(f_scan / max(total_frames, 1) * 100)}%)...",
                    model="Lucas-Kanade Optical Flow",
                    detail=f"Detected {len(detected_kfs)} candidate keyframes",
                )
        scan_cap.release()

        keyframe_indices = sorted(list(set(detected_kfs)))
        _set_stage(
            stage="Motion scan complete",
            detail=f"Found {len(keyframe_indices)} keyframes across {total_frames} frames",
            log_entry=f"Camera motion scan complete: detected {len(keyframe_indices)} benthic keyframes (indices: {', '.join(str(k) for k in keyframe_indices[:6])}{'...' if len(keyframe_indices) > 6 else ''})",
        )

        # 3. Setup Tracker Mode
        has_sam2 = False
        proxy_meta = {"scale_x": 1.0, "scale_y": 1.0}
        frames_dir = None

        if tracker_mode == "sam2":
            storage_dir = os.path.dirname(os.path.abspath(video_path))
            frames_dir = os.path.join(storage_dir, f"frames_{job_id}")
            sam2_tracker = SAM2BenthicTracker(device=opts.get("device_pref", "auto"))
            try:
                _set_stage(
                    stage="Extracting dual-resolution frames for SAM 2...",
                    model="SAM 2 Video Predictor",
                    action="Writing 640p proxy frames to disk",
                )
                _, proxy_meta = extract_video_frames_to_disk(video_path, frames_dir, max_dim=640, jpeg_quality=85)
                _set_stage(
                    stage="Initializing SAM 2 inference state...",
                    model="Meta SAM 2.1 Video Predictor",
                    action="Building memory feature bank",
                )
                sam2_tracker.init_video(frames_dir)
                has_sam2 = True
                _set_stage(
                    stage="SAM 2 Tracker active",
                    log_entry="Initialized SAM 2.1 Video Predictor with dual-resolution proxy",
                )
            except Exception as sam2_err:
                has_sam2 = False
                _set_stage(
                    stage="Optical Flow Tracker (Fallback)",
                    detail=f"SAM 2 unavailable ({str(sam2_err)}); using high-speed optical flow",
                    log_entry=f"SAM 2 initialization bypassed; operating with high-speed Optical Flow homography",
                )
        else:
            _set_stage(
                stage="High-Speed Optical Flow Benthic Tracker active",
                model="Lucas-Kanade Optical Flow Tracker",
                action="Homography coordinate warping at 60-120 FPS",
                log_entry="Initialized High-Speed Lucas-Kanade Benthic Colony Tracker",
            )

        with self._lock:
            job["keyframe_indices"] = list(keyframe_indices)
            job["keyframes_count"] = len(keyframe_indices)

        fps_clock = time.time()
        frames_processed_clock = 0

        def _record_frame_state(cur_f: int, current_active_tracks: List[Any]):
            nonlocal frames_processed_clock, fps_clock
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
            record = {
                "frame_index": cur_f,
                "timestamp_sec": timestamp_sec,
                "active_corals_count": len(current_active_tracks),
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
                job["active_corals_count"] = len(current_active_tracks)
                job["total_unique_corals"] = tracker._next_track_id - 1
                job["bleaching_prevalence_pct"] = round(bleach_pct, 1)
                job["frame_tracks_map"][cur_f] = list(current_active_tracks)
                job["time_series"].append(record)

            frames_processed_clock += 1
            now = time.time()
            if now - fps_clock >= 0.5:
                inst_fps = frames_processed_clock / max(now - fps_clock, 0.001)
                with self._lock:
                    job["fps_realtime"] = round(inst_fps, 1)
                fps_clock = now
                frames_processed_clock = 0

        try:
            scale_x = proxy_meta.get("scale_x", 1.0)
            scale_y = proxy_meta.get("scale_y", 1.0)

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
                    points_per_side = opts.get("points_per_side", 16)
                    _set_stage(
                        stage=f"Keyframe #{w_idx+1}/{len(keyframe_indices)} (Frame {kf_frame_idx}): Segmenting benthic colonies...",
                        model="CoralSCOP SAM ViT-B",
                        action=f"Prompt grid segmentation ({points_per_side}x{points_per_side} points)",
                        detail=f"Analyzing full-resolution frame {kf_frame_idx} ({width}x{height})",
                    )
                    masks_info, _ = run_segmentation(
                        model=sam_model,
                        image=kf_rgb,
                        points_per_side=points_per_side,
                        pred_iou_thresh=opts.get("pred_iou_thresh", 0.50),
                        stability_score_thresh=opts.get("stability_thresh", 0.50),
                        min_mask_region_area=100,
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
                        _set_stage(
                            stage=f"Keyframe #{w_idx+1}: Classifying {len(new_tracks)} new corals with BioCLIP & YOLO...",
                            model="BioCLIP ViT & YOLO11n Bleaching",
                            action="Taxonomy and bleaching health inference",
                            detail=f"Extracting {len(new_tracks)} bounding-box crops for fine-grained classification",
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

                                # Register newly discovered coral into SAM 2 tracker if active
                                if has_sam2 and tracker_mode == "sam2":
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

                            genera = sorted(list(set(t.taxonomy.get('genus', 'Coral') for t in new_tracks)))
                            _set_stage(
                                stage=f"Keyframe #{w_idx+1} cataloged {len(new_tracks)} new colonies",
                                detail=f"Dominant genera: {', '.join(genera[:3])}",
                                log_entry=f"Keyframe #{w_idx+1} (Frame {kf_frame_idx}): Segmented {len(masks_info)} corals, classified {len(new_tracks)} new colonies ({', '.join(genera[:3])})",
                            )
                    else:
                        _set_stage(
                            stage=f"Keyframe #{w_idx+1}: Re-associated {len(masks_info)} active colonies",
                            log_entry=f"Keyframe #{w_idx+1} (Frame {kf_frame_idx}): Re-associated {len(masks_info)} existing coral tracks",
                        )

                # Record keyframe state immediately into job state
                active_tracks = tracker.get_active_tracks()
                _record_frame_state(kf_frame_idx, active_tracks)

                # --- CONTINUOUS PROGRESSIVE STREAMING (Intervening Frames) ---
                if window_length > 1:
                    if tracker_mode == "sam2" and has_sam2 and sam2_tracker.total_tracks_count > 0:
                        _set_stage(
                            stage=f"Tracking frames {kf_frame_idx+1}-{next_kf_idx-1} via SAM 2...",
                            model="Meta SAM 2.1 Video Predictor",
                            action="Propagating neural deformable masks",
                            detail=f"Streaming {window_length - 1} frames across {sam2_tracker.total_tracks_count} colonies",
                        )
                        try:
                            for out_f, det_tracks in sam2_tracker.propagate_stream(
                                start_frame_idx=kf_frame_idx + 1,
                                num_frames=window_length - 1,
                                scale_x=scale_x,
                                scale_y=scale_y,
                            ):
                                if job["cancel_requested"]:
                                    break
                                for det in det_tracks:
                                    trk_id = det["track_id"]
                                    if trk_id in tracker._tracks:
                                        t = tracker._tracks[trk_id]
                                        t.bbox = det["bbox"]
                                        t.centroid = det["centroid"]
                                        t.polygon = det["polygon"]
                                        t.area_px = det["area_px"]
                                        t.mask = det["mask"]
                                        t.last_seen_frame = out_f
                                        t.time_since_update = 0
                                        t.history.append((det["bbox"], det["polygon"]))

                                active_tracks = tracker.get_active_tracks()
                                _record_frame_state(out_f, active_tracks)
                        except Exception:
                            # Fallback to homography warping on streaming error
                            for cur_f in range(kf_frame_idx + 1, next_kf_idx):
                                if job["cancel_requested"]:
                                    break
                                m_meta = motion_metas.get(cur_f, {})
                                step_H = m_meta.get("step_H", np.eye(3, dtype=np.float32))
                                tracker.warp_tracks_with_motion(
                                    step_H=step_H,
                                    frame_w=width,
                                    frame_h=height,
                                    frame_idx=cur_f,
                                )
                                active_tracks = tracker.get_active_tracks()
                                _record_frame_state(cur_f, active_tracks)
                    else:
                        _set_stage(
                            stage=f"Tracking frames {kf_frame_idx+1}-{next_kf_idx-1} via Optical Flow (60-120 FPS)...",
                            model="Lucas-Kanade Optical Flow Tracker",
                            action="High-speed homography motion warping",
                            detail=f"Propagating {len(active_tracks)} colonies across {window_length - 1} intervening frames",
                        )
                        for cur_f in range(kf_frame_idx + 1, next_kf_idx):
                            if job["cancel_requested"]:
                                break
                            m_meta = motion_metas.get(cur_f, {})
                            step_H = m_meta.get("step_H", np.eye(3, dtype=np.float32))
                            tracker.warp_tracks_with_motion(
                                step_H=step_H,
                                frame_w=width,
                                frame_h=height,
                                frame_idx=cur_f,
                            )
                            active_tracks = tracker.get_active_tracks()
                            _record_frame_state(cur_f, active_tracks)

            # Processing finished successfully
            if not job["cancel_requested"]:
                _set_stage(
                    stage="Exporting transect analytics, CSV, and overlay video...",
                    model="Transect Exporter",
                    action="Generating annotated MP4, CSV telemetry, and COCO video dataset",
                    detail="Writing survey deliverables to export cache",
                )
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

                _set_stage(
                    stage="Transect analysis completed",
                    model="None (Completed)",
                    action="Ready for playback & review",
                    detail=f"Tracked {tracker._next_track_id - 1} unique corals ({job['bleaching_prevalence_pct']}% bleaching)",
                    log_entry=f"Survey finished: {tracker._next_track_id - 1} unique corals, {job['bleaching_prevalence_pct']}% bleaching prevalence across {total_frames} frames",
                )

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
            if isinstance(frames_dir, str) and os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir, ignore_errors=True)
