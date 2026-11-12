"""
EfficientTAM (Experimental Fast Profile) Video Tracker Adapter for ReefVision.
Implements the BaseVideoTracker interface for fast offline video inference.
Provides actionable installation error when the optional upstream package is absent.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple, Iterator
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from core.device import get_optimal_device, check_torch_capabilities
from core.video.sam2_tracker import BaseVideoTracker


class EfficientTAMBenthicTracker(BaseVideoTracker):
    """
    Experimental fast-profile video tracker based on EfficientTAM.
    Provides ~6.5 FPS tracking at 512px on modest GPUs.
    """

    def __init__(
        self,
        device: str = "auto",
        model_id: str = "efficienttam_ti",
        model_size: int = 512,
        precision: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        self.device_pref = device
        self.model_id = model_id
        self.model_size = int(model_size)
        self.precision_pref = precision
        self.offload_to_cpu = offload_to_cpu

        self.device = "cpu"
        self.dtype = None
        self.predictor = None
        self.inference_state = None
        self.initialized = False
        self._video_source = None

        self._tracks_meta: Dict[int, Dict[str, Any]] = {}
        self._timings: Dict[str, float] = {
            "initialization_sec": 0.0,
            "prompt_registration_sec": 0.0,
            "propagation_sec": 0.0,
            "transfer_sec": 0.0,
            "geometry_sec": 0.0,
        }

    @property
    def total_tracks_count(self) -> int:
        return len(self._tracks_meta)

    def load(self):
        """Loads and prepares the EfficientTAM predictor once per worker."""
        if self.predictor is not None:
            return

        t0 = time.perf_counter()
        try:
            # Check for upstream EfficientTAM or adapted checkpoint
            try:
                import efficienttam
                from efficienttam import build_efficienttam_video_predictor
                self.predictor = build_efficienttam_video_predictor(self.model_id)
            except ImportError:
                # Check alternative import style or raise actionable instructions
                raise RuntimeError(
                    "EfficientTAM fast profile is not installed in the current environment.\n"
                    "To enable the experimental EfficientTAM profile, install from upstream:\n"
                    "  pip install git+https://github.com/facebookresearch/EfficientTAM.git\n"
                    "Or select the default reference model 'sam2' (SAM 2.1 Tiny)."
                )
        finally:
            self._timings["initialization_sec"] += time.perf_counter() - t0

    def init_video(self, video_source: str):
        """Initializes video inference state from frames directory."""
        self.load()
        self._video_source = video_source
        self.reset()
        t0 = time.perf_counter()
        try:
            if hasattr(self.predictor, "init_state"):
                self.inference_state = self.predictor.init_state(video_path=video_source)
            self.initialized = True
        finally:
            self._timings["initialization_sec"] += time.perf_counter() - t0

    def register_mask(
        self,
        frame_idx: int,
        obj_id: int,
        binary_mask: np.ndarray,
        is_correction: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registers a ground-truth or keyframe-discovered binary mask for tracking."""
        if not self.initialized:
            raise RuntimeError("Cannot register mask before init_video() has been called.")

        t0 = time.perf_counter()
        try:
            mask_bool = binary_mask.astype(bool)
            if self.model_size != 1024 and _HAS_CV2:
                # Resize to tracker proxy resolution
                h, w = mask_bool.shape[:2]
                if max(h, w) > self.model_size:
                    scale = self.model_size / float(max(h, w))
                    tw = max(16, int(round(w * scale)))
                    th = max(16, int(round(h * scale)))
                    mask_bool = cv2.resize(mask_bool.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)

            if hasattr(self.predictor, "add_new_mask"):
                self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_bool,
                )

            self._tracks_meta[obj_id] = {
                "track_id": obj_id,
                "first_seen_frame": frame_idx,
                "is_correction": is_correction,
                "metadata": metadata or {},
            }
        finally:
            self._timings["prompt_registration_sec"] += time.perf_counter() - t0

    def propagate(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """
        Streaming generator yielding (frame_idx, detections) for [start_frame_idx, end_frame_idx).
        """
        if not self.initialized:
            raise RuntimeError("Tracker must be initialized before propagation.")

        if start_frame_idx >= end_frame_idx:
            return

        expected_count = end_frame_idx - start_frame_idx
        if not self._tracks_meta:
            for f in range(start_frame_idx, end_frame_idx):
                yield f, []
            return

        t0 = time.perf_counter()
        returned_indices = set()
        try:
            prop_gen = self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=expected_count - 1,
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in prop_gen:
                if out_frame_idx >= end_frame_idx:
                    break
                returned_indices.add(out_frame_idx)

                frame_detections = []
                for idx, obj_id in enumerate(out_obj_ids):
                    if obj_id not in self._tracks_meta:
                        continue
                    m_logits = out_mask_logits[idx]
                    bin_mask = (m_logits > 0.0)
                    if hasattr(bin_mask, "cpu"):
                        bin_mask = bin_mask.cpu().numpy()
                    if bin_mask.ndim == 3:
                        bin_mask = bin_mask[0]

                    # Extract canonical geometry
                    geom = self._extract_geometry(bin_mask, scale_x, scale_y)
                    if geom:
                        frame_detections.append({
                            "track_id": int(obj_id),
                            **geom,
                            **self._tracks_meta[obj_id].get("metadata", {}),
                        })

                yield out_frame_idx, frame_detections

            for f_fill in range(start_frame_idx, end_frame_idx):
                if f_fill not in returned_indices:
                    yield f_fill, []
        finally:
            self._timings["propagation_sec"] += time.perf_counter() - t0

    def _extract_geometry(self, mask: np.ndarray, scale_x: float, scale_y: float) -> Optional[Dict[str, Any]]:
        """Extracts canonical polygon, bbox, centroid, and area_px from a binary mask."""
        if not _HAS_CV2:
            return None
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area_proxy = cv2.contourArea(largest)
        if area_proxy < 4:
            return None

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.015 * peri, True)
        polygon = [
            [int(round(pt[0][0] * scale_x)), int(round(pt[0][1] * scale_y))]
            for pt in approx
        ]
        if len(polygon) < 3:
            return None

        bx, by, bw, bh = cv2.boundingRect(largest)
        bbox = [
            int(round(bx * scale_x)),
            int(round(by * scale_y)),
            max(1, int(round(bw * scale_x))),
            max(1, int(round(bh * scale_y))),
        ]

        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(round((M["m10"] / M["m00"]) * scale_x))
            cy = int(round((M["m01"] / M["m00"]) * scale_y))
        else:
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2

        area_px = int(round(area_proxy * scale_x * scale_y))
        return {
            "bbox": bbox,
            "polygon": polygon,
            "centroid": (cx, cy),
            "area_px": area_px,
        }

    def retire_objects(self, obj_ids: List[int]):
        """Retires objects from active state."""
        for oid in obj_ids:
            self._tracks_meta.pop(oid, None)

    def reset(self):
        """Resets active tracks and inference state."""
        self._tracks_meta.clear()
        if hasattr(self.predictor, "reset_state") and self.inference_state is not None:
            try:
                self.predictor.reset_state(self.inference_state)
            except Exception:
                pass

    def close(self):
        """Releases memory and closes predictor."""
        self.reset()
        self.predictor = None
        self.inference_state = None
        self.initialized = False

    def get_timings(self) -> Dict[str, float]:
        return dict(self._timings)
