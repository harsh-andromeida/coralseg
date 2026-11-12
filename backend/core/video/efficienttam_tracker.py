"""
EfficientTAM Benthic Video Tracker Adapter for ReefVision.
Integrates the upstream EfficientTAM (Efficient Track Anything) model
as an explicit experimental fast profile conforming to BaseVideoTracker.
Upstream repository: https://github.com/yformer/EfficientTAM
Tested baseline commit: abcd061ebd3cc6e7527d152d75b890126aaa53f6
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

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from core.video.sam2_tracker import BaseVideoTracker

try:
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
except ImportError:
    try:
        from efficienttam.build_efficienttam import build_efficienttam_video_predictor
    except ImportError:
        build_efficienttam_video_predictor = None


class EfficientTAMBenthicTracker(BaseVideoTracker):
    """
    Experimental fast video tracker adapter wrapping EfficientTAM.
    Accepts 512px or 1024px inputs, handles memory offloading, and exposes identical
    adapter methods (register_mask, propagate, retire_objects, close).
    """

    def __init__(
        self,
        device: str = "auto",
        model_size: int = 512,
        precision: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        self.requested_device = device
        self.model_size = 512 if model_size == 512 else 1024
        self.precision_str = precision
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.offload_to_cpu = offload_to_cpu

        # Device resolution
        if device == "auto":
            if _HAS_TORCH and torch.cuda.is_available():
                self.device = "cuda"
            elif _HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Precision resolution
        if self.device == "cuda":
            if precision == "bf16" and _HAS_TORCH and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16 if _HAS_TORCH else None
        else:
            self.dtype = torch.float32 if _HAS_TORCH else None

        self.predictor = None
        self.inference_state = None
        self._tracks_meta: Dict[int, Dict[str, Any]] = {}
        self._video_source: Optional[str] = None
        self.initialized = False

        self._timings = {
            "initialization_sec": 0.0,
            "prompt_registration_sec": 0.0,
            "propagation_sec": 0.0,
            "transfer_sec": 0.0,
            "geometry_sec": 0.0,
        }

    @property
    def total_tracks_count(self) -> int:
        return len(self._tracks_meta)

    def _resolve_config_and_checkpoint(self) -> Tuple[str, str]:
        """Resolves configuration YAML and checkpoint weights path."""
        # 1. Config path
        if self.config_path and os.path.exists(self.config_path):
            cfg = self.config_path
        else:
            default_cfg = f"configs/efficienttam/efficienttam_ti_{self.model_size}x{self.model_size}.yaml"
            cfg = default_cfg

        # 2. Checkpoint path
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ckpt = self.checkpoint_path
        else:
            candidates = [
                f"checkpoints/efficienttam_ti_{self.model_size}x{self.model_size}.pt",
                f"models/efficienttam_ti_{self.model_size}x{self.model_size}.pt",
                os.path.join(os.path.dirname(__file__), "..", "..", "models", f"efficienttam_ti_{self.model_size}x{self.model_size}.pt"),
                os.path.expanduser(f"~/.cache/efficienttam/efficienttam_ti_{self.model_size}x{self.model_size}.pt"),
            ]
            ckpt = None
            for c in candidates:
                if os.path.exists(c):
                    ckpt = c
                    break
            if ckpt is None:
                ckpt = candidates[0]

        return cfg, ckpt

    def load(self):
        """Loads and prepares the EfficientTAM predictor once per worker."""
        if self.predictor is not None:
            return

        t0 = time.perf_counter()
        try:
            if build_efficienttam_video_predictor is None:
                raise RuntimeError(
                    "EfficientTAM fast profile is not installed in the current environment.\n"
                    "To enable the experimental EfficientTAM profile, install the upstream package:\n"
                    "  pip install git+https://github.com/yformer/EfficientTAM.git@abcd061ebd3cc6e7527d152d75b890126aaa53f6\n"
                    "Or select the default reference model 'sam2' (SAM 2.1 Tiny)."
                )

            cfg, ckpt = self._resolve_config_and_checkpoint()
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(
                    f"EfficientTAM checkpoint not found at: {ckpt}\n"
                    f"Download efficienttam_ti_{self.model_size}x{self.model_size}.pt from "
                    f"https://github.com/yformer/EfficientTAM and place it in checkpoints/ or models/."
                )

            self.predictor = build_efficienttam_video_predictor(
                config_file=cfg,
                ckpt_path=ckpt,
                device=self.device,
            )
        finally:
            self._timings["initialization_sec"] += time.perf_counter() - t0

    def init_video(self, video_source: str, frame_offset: int = 0):
        """Initializes video inference state from frames directory."""
        self.load()
        self._video_source = video_source
        self.reset()
        self.frame_offset = frame_offset
        t0 = time.perf_counter()
        try:
            if hasattr(self.predictor, "init_state"):
                self.inference_state = self.predictor.init_state(
                    video_path=video_source,
                    offload_video_to_cpu=self.offload_to_cpu,
                    offload_state_to_cpu=self.offload_to_cpu,
                )
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
                h, w = mask_bool.shape[:2]
                if max(h, w) > self.model_size:
                    scale = self.model_size / float(max(h, w))
                    tw = max(16, int(round(w * scale)))
                    th = max(16, int(round(h * scale)))
                    mask_bool = cv2.resize(mask_bool.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)

            local_frame_idx = frame_idx - getattr(self, "frame_offset", 0)
            if hasattr(self.predictor, "add_new_mask"):
                self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=local_frame_idx,
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
        Streaming generator yielding (frame_idx, detections) for half-open range [start_frame_idx, end_frame_idx).
        Measures pure GPU propagation latency before advancing generator, transfers masks,
        extracts canonical geometry, and asserts exact complete frame coverage.
        """
        if not self.initialized:
            raise RuntimeError("Tracker must be initialized before propagation.")

        if start_frame_idx >= end_frame_idx:
            return

        expected_indices = list(range(start_frame_idx, end_frame_idx))
        if not self._tracks_meta:
            for f in expected_indices:
                yield f, []
            return

        expected_count = end_frame_idx - start_frame_idx
        offset = getattr(self, "frame_offset", 0)
        local_start = start_frame_idx - offset
        frames_yielded = 0

        prop_gen = self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=local_start,
            max_frame_num_to_track=expected_count - 1,
        )

        try:
            it = iter(prop_gen)
            while frames_yielded < len(expected_indices):
                t_infer_start = time.perf_counter()
                try:
                    out_frame_idx, out_obj_ids, out_mask_logits = next(it)
                except StopIteration:
                    break
                t_infer_end = time.perf_counter()
                self._timings["propagation_sec"] += (t_infer_end - t_infer_start)

                abs_out_frame_idx = out_frame_idx + offset
                if abs_out_frame_idx >= end_frame_idx:
                    break

                expected_idx = expected_indices[frames_yielded]
                if abs_out_frame_idx != expected_idx:
                    raise RuntimeError(
                        f"EfficientTAM propagation index mismatch: expected {expected_idx}, got {abs_out_frame_idx}"
                    )

                if len(out_obj_ids) == 0:
                    yield abs_out_frame_idx, []
                    frames_yielded += 1
                    continue

                t_trans = time.perf_counter()
                if hasattr(out_mask_logits, "cpu"):
                    bin_masks = (out_mask_logits > 0.0).cpu().numpy()
                else:
                    bin_masks = (out_mask_logits > 0.0)
                if bin_masks.ndim == 2:
                    bin_masks = np.expand_dims(bin_masks, axis=0)
                self._timings["transfer_sec"] += time.perf_counter() - t_trans

                t_geom = time.perf_counter()
                frame_detections = []
                for idx, obj_id in enumerate(out_obj_ids):
                    if obj_id not in self._tracks_meta:
                        continue
                    bin_mask = bin_masks[idx]
                    if bin_mask.ndim == 3:
                        bin_mask = bin_mask[0]

                    geom = self._extract_geometry(bin_mask, scale_x, scale_y)
                    if geom:
                        frame_detections.append({
                            "track_id": int(obj_id),
                            "id": int(obj_id),
                            "id_str": f"#{obj_id}",
                            **geom,
                            **self._tracks_meta[obj_id].get("metadata", {}),
                        })

                self._timings["geometry_sec"] += time.perf_counter() - t_geom

                yield abs_out_frame_idx, frame_detections
                frames_yielded += 1

            if frames_yielded != len(expected_indices):
                raise RuntimeError(
                    f"EfficientTAM propagation truncated: expected {len(expected_indices)} frames "
                    f"[{start_frame_idx}, {end_frame_idx}), yielded only {frames_yielded}"
                )
        finally:
            if hasattr(prop_gen, "close"):
                prop_gen.close()

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
            "area": area_px,
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
