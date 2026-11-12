import cv2
import numpy as np
import torch
import contextlib
from typing import Dict, List, Any, Optional, Tuple


def _resolve_torch_device(pref: Optional[str] = None) -> str:
    """Selects best available torch device based on preference and hardware."""
    if pref in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if pref == "cpu":
        return "cpu"
    # Auto
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast_context(device: str):
    """Provides hardware-optimized mixed precision context for CUDA (bfloat16/float16)."""
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


class SAM2BenthicTracker:
    """
    Continuous benthic colony tracking using Meta's SAM 2.1 Video Predictor.
    Optimized for NVIDIA CUDA Tensor Cores with automatic mixed precision (bfloat16/float16),
    while preserving smooth PyTorch MPS and CPU execution for testing.
    """

    def __init__(self, device: Optional[str] = None, model_id: str = "facebook/sam2.1-hiera-tiny"):
        self.device = _resolve_torch_device(device)
        self.model_id = model_id
        self._predictor = None
        self._inference_state = None
        self._tracks_meta: Dict[int, Dict[str, Any]] = {}
        self.initialized = False

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

    def load(self):
        if self._predictor is not None:
            return
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        self._predictor = SAM2VideoPredictor.from_pretrained(self.model_id, device=self.device)

    def init_video(self, frames_dir: str):
        """Initializes the SAM 2 video predictor with extracted 25 FPS JPEG frames."""
        self.load()
        with torch.inference_mode(), _autocast_context(self.device):
            # offload_video_to_cpu=True and offload_state_to_cpu=True prevent SAM 2 from eating
            # all GPU VRAM when propagating through video frames in constrained environments like Colab.
            try:
                self._inference_state = self._predictor.init_state(
                    video_path=frames_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
            except TypeError:
                # Fallback for older sam2 versions without offload parameters
                self._inference_state = self._predictor.init_state(video_path=frames_dir)

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._tracks_meta.clear()
        self.initialized = True

    def register_coral(
        self,
        frame_idx: int,
        track_id: int,
        binary_mask: np.ndarray,
        taxonomy: Dict[str, Any],
        bleaching: Dict[str, Any],
        color_hex: str,
        predicted_iou: float = 0.85,
    ):
        """
        Registers a new coral colony into the SAM 2 inference state at frame_idx.
        """
        if not self.initialized:
            raise RuntimeError("SAM2BenthicTracker not initialized with a video.")

        # Ensure mask is bool or uint8 2D
        mask_np = binary_mask.astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        # Resample mask to match SAM 2 video state resolution if proxy downscaling was used
        state_h = self._inference_state.get("video_height", mask_np.shape[0])
        state_w = self._inference_state.get("video_width", mask_np.shape[1])
        if (mask_np.shape[0] != state_h or mask_np.shape[1] != state_w) and state_h > 0 and state_w > 0:
            mask_resized = cv2.resize(mask_np.astype(np.uint8), (state_w, state_h), interpolation=cv2.INTER_NEAREST)
            mask_np = mask_resized.astype(bool)

        # Add mask prompt to SAM 2 under inference context
        with torch.inference_mode(), _autocast_context(self.device):
            self._predictor.add_new_mask(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=track_id,
                mask=mask_np,
            )

        cond_label = "Bleached" if bleaching.get("is_bleached", False) else "Healthy"
        self._tracks_meta[track_id] = {
            "track_id": track_id,
            "genus": taxonomy.get("genus", "Coral"),
            "growth_form": taxonomy.get("growth_form", "-"),
            "taxon_conf": round(float(taxonomy.get("confidence_pct", 85.0)), 1),
            "condition": cond_label,
            "condition_conf": round(float(bleaching.get("confidence_pct", 90.0)), 1),
            "is_bleached": bleaching.get("is_bleached", False),
            "color_hex": color_hex,
            "first_seen_frame": frame_idx,
            "predicted_iou": round(float(predicted_iou), 3),
        }

    def propagate_stream(
        self,
        start_frame_idx: int,
        num_frames: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ):
        """
        Streaming generator that yields (out_frame_idx, frame_tracks) immediately as SAM 2
        computes each frame's memory-attention propagation.
        Applies affine scaling (scale_x, scale_y) to remap coordinates back to raw video pixels.
        Batches GPU-to-CPU memory transfers to eliminate synchronization bottlenecks.
        """
        if not self.initialized or not self._tracks_meta:
            return

        with torch.inference_mode(), _autocast_context(self.device):
            for out_frame_idx, out_obj_ids, out_mask_logits in self._predictor.propagate_in_video(
                self._inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=num_frames,
            ):
                if len(out_obj_ids) == 0:
                    yield out_frame_idx, []
                    continue

                # Batch transfer all object masks to CPU in a single tensor operation
                masks_bool = (out_mask_logits > 0.0).squeeze(1).cpu().numpy()
                if masks_bool.ndim == 2:
                    masks_bool = np.expand_dims(masks_bool, axis=0)

                frame_tracks = []
                for i, obj_id in enumerate(out_obj_ids):
                    if obj_id not in self._tracks_meta:
                        continue

                    meta = self._tracks_meta[obj_id]
                    mask_bool = masks_bool[i]
                    if mask_bool.ndim == 3:
                        mask_bool = mask_bool[0]

                    area_px = int(np.sum(mask_bool))
                    if area_px < 15:
                        continue

                    # Extract contour polygons & bounding box using fast compiled OpenCV
                    contours, _ = cv2.findContours(
                        mask_bool.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    poly = []
                    bbox = [0, 0, 0, 0]
                    cx, cy = 0, 0
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if len(c) >= 3:
                            poly = c[:, 0, :].tolist()
                        bx, by, bw, bh = cv2.boundingRect(c)
                        bbox = [int(bx), int(by), max(1, int(bw)), max(1, int(bh))]
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                    else:
                        continue

                    # Remap proxy coordinates back to raw video display coordinates
                    if scale_x != 1.0 or scale_y != 1.0:
                        poly = [[int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))] for pt in poly]
                        bbox = [
                            int(round(bbox[0] * scale_x)),
                            int(round(bbox[1] * scale_y)),
                            max(1, int(round(bbox[2] * scale_x))),
                            max(1, int(round(bbox[3] * scale_y))),
                        ]
                        cx = int(round(cx * scale_x))
                        cy = int(round(cy * scale_y))
                        area_px = int(round(area_px * scale_x * scale_y))

                    frame_tracks.append({
                        "track_id": obj_id,
                        "id": obj_id,
                        "id_str": f"#{obj_id}",
                        "genus": meta["genus"],
                        "growth_form": meta["growth_form"],
                        "condition": meta["condition"],
                        "condition_conf": meta["condition_conf"],
                        "taxon_conf": meta["taxon_conf"],
                        "is_bleached": meta["is_bleached"],
                        "color_hex": meta["color_hex"],
                        "area_px": area_px,
                        "area": area_px,
                        "predicted_iou": meta["predicted_iou"],
                        "bbox": bbox,
                        "centroid": [cx, cy],
                        "polygon": poly,
                        "mask": mask_bool,
                    })

                yield out_frame_idx, frame_tracks

    def propagate_window(
        self,
        start_frame_idx: int,
        num_frames: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Propagates SAM 2 tracker from start_frame_idx for num_frames frames.
        Returns a dict: frame_idx -> List of active track detection dicts.
        """
        results: Dict[int, List[Dict[str, Any]]] = {}
        for f_idx, f_tracks in self.propagate_stream(
            start_frame_idx=start_frame_idx,
            num_frames=num_frames,
            scale_x=scale_x,
            scale_y=scale_y,
        ):
            results[f_idx] = f_tracks
        return results

    def get_track_metadata(self, track_id: int) -> Optional[Dict[str, Any]]:
        return self._tracks_meta.get(track_id)

    @property
    def total_tracks_count(self) -> int:
        return len(self._tracks_meta)
