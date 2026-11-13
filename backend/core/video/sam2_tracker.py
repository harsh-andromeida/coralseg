"""
SAM 2.1 Video Predictor Adapter for Marine Benthic Colony Tracking.
Implements the BaseVideoTracker adapter boundary with explicit precision selection,
half-open range propagation [start, end), object retirement, separate stage timings,
and zero routine empty_cache() calls during normal execution.
"""

import abc
import time
import queue
import threading
import contextlib
import cv2
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Tuple, Iterator, Union

from core.device import check_torch_capabilities


class BaseVideoTracker(abc.ABC):
    """
    Standard tracker adapter boundary for video predictors (SAM 2, EfficientTAM).
    Decouples lifecycle, mask registration, half-open propagation [start, end), and object retirement.
    """

    @abc.abstractmethod
    def load(self):
        """Loads and prepares predictor once per worker."""
        pass

    @abc.abstractmethod
    def init_video(self, video_source: str):
        """Initializes video inference state from frames directory or video path."""
        pass

    @abc.abstractmethod
    def register_mask(
        self,
        frame_idx: int,
        obj_id: int,
        binary_mask: np.ndarray,
        is_correction: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registers or updates a mask for obj_id at frame_idx independently of classification."""
        pass

    @abc.abstractmethod
    def propagate(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """
        Streaming generator yielding (frame_idx, frame_detections) for half-open range [start, end).
        Guarantees exact returned frame indices.
        """
        pass

    @abc.abstractmethod
    def retire_objects(self, obj_ids: List[int]):
        """Retires objects so they no longer consume active inference memory."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Resets active tracking state while retaining loaded model."""
        pass

    @abc.abstractmethod
    def close(self):
        """Releases all inference states and memory."""
        pass

    @abc.abstractmethod
    def get_timings(self) -> Dict[str, float]:
        """Returns separate timings for initialization, registration, propagation, transfer, and geometry."""
        pass


def _resolve_torch_device(pref: Optional[str] = None) -> str:
    """Selects best available torch device based on preference and hardware."""
    caps = check_torch_capabilities()
    if pref in ("cuda", "gpu") and caps["cuda_available"]:
        return "cuda"
    if pref == "mps" and caps["mps_available"]:
        return "mps"
    if pref == "cpu":
        return "cpu"
    # Auto
    if caps["cuda_available"]:
        return "cuda"
    if caps["mps_available"]:
        return "mps"
    return "cpu"


def _sync_device(device: str):
    """Synchronizes accelerator stream so host timings reflect true hardware compute latency."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _resolve_precision(device: str, user_precision: Optional[str] = None) -> Tuple[torch.dtype, bool]:
    """
    Selects hardware-validated precision:
    - CUDA: BF16 only when native without emulation; otherwise FP16.
    - MPS: FP32 default for correctness; FP16 as tested profile.
    - CPU: FP32.
    Returns (torch_dtype, use_autocast).
    """
    caps = check_torch_capabilities()

    if device == "cuda" and caps["cuda_available"]:
        native_bf16 = caps["native_precision"]["cuda_bf16_native"]
        if user_precision in ("bfloat16", "bf16"):
            dtype = torch.bfloat16 if native_bf16 else torch.float16
        elif user_precision in ("float16", "fp16"):
            dtype = torch.float16
        elif user_precision in ("float32", "fp32"):
            return torch.float32, False
        else:
            dtype = torch.bfloat16 if native_bf16 else torch.float16
        return dtype, True

    if device == "mps" and caps["mps_available"]:
        # MPS default is FP32 for verified numerical stability
        if user_precision in ("float16", "fp16"):
            return torch.float16, True
        return torch.float32, False

    # CPU
    return torch.float32, False


def _get_autocast_context(device: str, dtype: torch.dtype, enabled: bool):
    """Builds appropriate mixed precision context."""
    if enabled and device == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=dtype)
    if enabled and device == "mps":
        try:
            return torch.autocast("mps", dtype=dtype)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


class SAM2BenthicTracker(BaseVideoTracker):
    """
    Continuous benthic colony tracking using Meta's SAM 2.1 Video Predictor.
    Implements BaseVideoTracker adapter boundary.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        model_size: int = 1024,
        precision: Optional[str] = None,
        vos_optimized: bool = False,
        compile_tracker: bool = False,
        offload_to_cpu: bool = True,
    ):
        self.device = _resolve_torch_device(device)
        self.model_id = model_id
        self.model_size = int(model_size)
        self.precision_pref = precision
        self.compile_tracker = bool(compile_tracker or vos_optimized)
        # vos_optimized / torch.compile is qualified only on CUDA; MPS & CPU enforce eager
        self.vos_optimized = bool(self.compile_tracker and self.device == "cuda")
        self.offload_to_cpu = offload_to_cpu

        self.dtype, self.use_autocast = _resolve_precision(self.device, precision)

        self._predictor = None
        self._inference_state = None
        self._tracks_meta: Dict[int, Dict[str, Any]] = {}
        self.initialized = False
        self.recompilations_count = 0

        # Timing instrumentation
        self._timings: Dict[str, float] = {
            "initialization_sec": 0.0,
            "prompt_registration_sec": 0.0,
            "propagation_sec": 0.0,
            "transfer_sec": 0.0,
            "geometry_sec": 0.0,
            "compile_warmup_sec": 0.0,
            "queue_wait_sec": 0.0,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

    def _run_compile_warmup(self):
        """Executes a 1-frame dummy warm-up forward pass to trigger ahead-of-time graph compilation."""
        if not (self.vos_optimized and self.device == "cuda"):
            return
        t_start = time.perf_counter()
        try:
            img_size = getattr(self._predictor, "image_size", self.model_size)
            dummy_images = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32, device="cpu" if self.offload_to_cpu else self.device)
            dummy_state = {
                "images": dummy_images,
                "num_frames": 1,
                "offload_video_to_cpu": self.offload_to_cpu,
                "offload_state_to_cpu": self.offload_to_cpu,
                "video_height": img_size,
                "video_width": img_size,
                "device": self._predictor.device,
                "storage_device": torch.device("cpu") if self.offload_to_cpu else self._predictor.device,
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": OrderedDict(),
                "obj_idx_to_id": OrderedDict(),
                "obj_ids": [],
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "frames_tracked_per_obj": {},
            }
            with torch.inference_mode(), _get_autocast_context(self.device, self.dtype, self.use_autocast):
                # 1. Warm up and compile visual backbone
                self._predictor._get_image_feature(dummy_state, frame_idx=0, batch_size=1)
                # 2. Warm up mask prompt encoder & mask decoder
                dummy_mask = np.zeros((img_size, img_size), dtype=bool)
                dummy_mask[img_size // 4 : img_size // 2, img_size // 4 : img_size // 2] = True
                self._predictor.add_new_mask(dummy_state, frame_idx=0, obj_id=1, mask=dummy_mask)
                # 3. Warm up memory encoder and attention
                gen = self._predictor.propagate_in_video(dummy_state, start_frame_idx=0, max_frame_num_to_track=0)
                for _ in gen:
                    pass
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
            self._timings["compile_warmup_sec"] = round(time.perf_counter() - t_start, 4)
        except Exception as e:
            print(f"[SAM2BenthicTracker] Compile warm-up notice: {e}")
            self._timings["compile_warmup_sec"] = round(time.perf_counter() - t_start, 4)

    def load(self):
        """Builds predictor once per worker and reuses it."""
        if self._predictor is not None:
            return

        t0 = time.perf_counter()
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        kwargs: Dict[str, Any] = {"device": self.device}
        if self.model_size == 512:
            kwargs["hydra_overrides_extra"] = ["++model.image_size=512"]

        if self.vos_optimized and self.device == "cuda":
            kwargs["vos_optimized"] = True

        self._predictor = SAM2VideoPredictor.from_pretrained(self.model_id, **kwargs)

        if self.vos_optimized and self.device == "cuda":
            self._run_compile_warmup()

        self._timings["initialization_sec"] += time.perf_counter() - t0

    def _init_state_from_memory(
        self,
        frames: Union[List[np.ndarray], torch.Tensor],
        orig_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Directly initializes SAM 2 inference state from in-memory frames without disk I/O."""
        img_size = getattr(self._predictor, "image_size", self.model_size)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if isinstance(frames, torch.Tensor):
            images = frames
            v_h, v_w = orig_hw if orig_hw else (images.shape[-2], images.shape[-1])
        else:
            if len(frames) == 0:
                raise ValueError("Frame list is empty")
            first_f = frames[0]
            v_h = orig_hw[0] if orig_hw else first_f.shape[0]
            v_w = orig_hw[1] if orig_hw else first_f.shape[1]

            tensors = []
            for f in frames:
                if f.shape[0] != img_size or f.shape[1] != img_size:
                    f_res = cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                else:
                    f_res = f
                # Ensure RGB float32 [0, 1]
                t = torch.from_numpy(f_res).permute(2, 0, 1).float() / 255.0
                tensors.append(t)
            images = torch.stack(tensors)
            images = (images - mean) / std

        target_device = "cpu" if self.offload_to_cpu else self.device
        images = images.to(target_device)
        return self._build_inference_state(images, v_h, v_w)

    def _build_inference_state(self, images: torch.Tensor, v_h: int, v_w: int) -> Dict[str, Any]:
        inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": self.offload_to_cpu,
            "offload_state_to_cpu": self.offload_to_cpu,
            "video_height": v_h,
            "video_width": v_w,
            "device": self._predictor.device,
            "storage_device": torch.device("cpu") if self.offload_to_cpu else self._predictor.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        self._predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def init_video(
        self,
        video_source: Any,
        frame_offset: int = 0,
        orig_hw: Optional[Tuple[int, int]] = None,
    ):
        """
        Initializes SAM 2 inference state from frames directory OR in-memory frame buffer.
        """
        t0 = time.perf_counter()
        self.load()
        self.reset()
        self.frame_offset = frame_offset

        with torch.inference_mode(), _get_autocast_context(self.device, self.dtype, self.use_autocast):
            if isinstance(video_source, (list, tuple)) or isinstance(video_source, torch.Tensor):
                self._inference_state = self._init_state_from_memory(video_source, orig_hw=orig_hw)
            elif hasattr(video_source, "get_chunk_frames"):
                frames = video_source.get_chunk_frames(frame_offset, frame_offset + getattr(video_source, "chunk_size", 128))
                self._inference_state = self._init_state_from_memory(frames, orig_hw=orig_hw or (video_source.proxy_h, video_source.proxy_w))
            elif isinstance(video_source, str):
                try:
                    self._inference_state = self._predictor.init_state(
                        video_path=video_source,
                        offload_video_to_cpu=self.offload_to_cpu,
                        offload_state_to_cpu=self.offload_to_cpu,
                    )
                except TypeError:
                    self._inference_state = self._predictor.init_state(video_path=video_source)
            else:
                raise TypeError(f"Unsupported video_source type: {type(video_source)}")

        self._tracks_meta.clear()
        self.initialized = True
        self._timings["initialization_sec"] += time.perf_counter() - t0

    def register_mask(
        self,
        frame_idx: int,
        obj_id: int,
        binary_mask: np.ndarray,
        is_correction: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Registers a mask prompt independently of classification.
        Supports both new object registration and keyframe mask corrections.
        """
        if not self.initialized:
            raise RuntimeError("SAM2BenthicTracker not initialized with a video.")

        t0 = time.perf_counter()

        mask_np = binary_mask.astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        # Resample mask to match SAM 2 video state resolution if proxy downscaling was used
        state_h = self._inference_state.get("video_height", mask_np.shape[0])
        state_w = self._inference_state.get("video_width", mask_np.shape[1])
        if (mask_np.shape[0] != state_h or mask_np.shape[1] != state_w) and state_h > 0 and state_w > 0:
            mask_resized = cv2.resize(
                mask_np.astype(np.uint8),
                (state_w, state_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_np = mask_resized.astype(bool)

        local_frame_idx = frame_idx - getattr(self, "frame_offset", 0)

        with torch.inference_mode(), _get_autocast_context(self.device, self.dtype, self.use_autocast):
            self._predictor.add_new_mask(
                inference_state=self._inference_state,
                frame_idx=local_frame_idx,
                obj_id=obj_id,
                mask=mask_np,
            )

        if metadata:
            meta = dict(metadata)
        else:
            meta = self._tracks_meta.get(obj_id, {})

        meta.setdefault("track_id", obj_id)
        meta.setdefault("genus", "Coral")
        meta.setdefault("growth_form", "-")
        meta.setdefault("taxon_conf", 85.0)
        meta.setdefault("condition", "Healthy")
        meta.setdefault("condition_conf", 90.0)
        meta.setdefault("is_bleached", False)
        meta.setdefault("color_hex", "#00ffff")
        meta.setdefault("first_seen_frame", frame_idx)
        meta.setdefault("predicted_iou", 0.85)
        meta["last_corrected_frame"] = frame_idx

        self._tracks_meta[obj_id] = meta
        self._timings["prompt_registration_sec"] += time.perf_counter() - t0

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
        """Backwards compatibility helper mapping to register_mask."""
        cond_label = "Bleached" if bleaching.get("is_bleached", False) else "Healthy"
        meta = {
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
        self.register_mask(
            frame_idx=frame_idx,
            obj_id=track_id,
            binary_mask=binary_mask,
            is_correction=(track_id in self._tracks_meta),
            metadata=meta,
        )

    def propagate(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """
        Propagates SAM 2 across half-open range [start_frame_idx, end_frame_idx).
        Translates to SAM 2's inclusive max_frame_num_to_track and asserts exact frame coverage.
        """
        if not self.initialized:
            return

        expected_count = end_frame_idx - start_frame_idx
        if expected_count <= 0:
            return

        expected_frame_indices = list(range(start_frame_idx, end_frame_idx))

        if not self._tracks_meta:
            # No colonies currently registered; yield empty tracks for exact frame coverage
            for f in expected_frame_indices:
                yield f, []
            return
        # SAM 2 propagate_in_video uses inclusive bounds where max_frame_num_to_track is frames after start
        offset = getattr(self, "frame_offset", 0)
        local_start = start_frame_idx - offset
        sam_max_frames = (end_frame_idx - 1) - start_frame_idx
        frames_yielded = 0

        frame_queue = queue.Queue(maxsize=2)
        worker_error = []
        cancel_event = threading.Event()

        def _inference_worker():
            gen = None
            try:
                with torch.inference_mode(), _get_autocast_context(self.device, self.dtype, self.use_autocast):
                    gen = self._predictor.propagate_in_video(
                        self._inference_state,
                        start_frame_idx=local_start,
                        max_frame_num_to_track=sam_max_frames,
                    )
                    t_infer_prev = time.perf_counter()
                    for out_frame_idx, out_obj_ids, out_mask_logits in gen:
                        if cancel_event.is_set():
                            break

                        t_now = time.perf_counter()
                        self._timings["propagation_sec"] += (t_now - t_infer_prev)

                        t_trans_start = time.perf_counter()
                        masks_bool = None
                        if len(out_obj_ids) > 0:
                            # Transfer masks to host; .cpu() performs required stream synchronization
                            masks_bool = (out_mask_logits > 0.0).squeeze(1).cpu().numpy()
                            if masks_bool.ndim == 2:
                                masks_bool = np.expand_dims(masks_bool, axis=0)
                        self._timings["transfer_sec"] += (time.perf_counter() - t_trans_start)

                        item = (out_frame_idx, list(out_obj_ids), masks_bool)
                        put_ok = False
                        while not cancel_event.is_set():
                            try:
                                frame_queue.put(item, timeout=0.1)
                                put_ok = True
                                break
                            except queue.Full:
                                continue
                        if not put_ok or cancel_event.is_set():
                            break
                        t_infer_prev = time.perf_counter()
            except Exception as exc:
                worker_error.append(exc)
            finally:
                if gen is not None and hasattr(gen, "close"):
                    try:
                        gen.close()
                    except Exception:
                        pass
                try:
                    frame_queue.put(None, timeout=0.5)
                except Exception:
                    pass

        worker_thread = threading.Thread(target=_inference_worker, daemon=True)
        worker_thread.start()

        try:
            while frames_yielded < len(expected_frame_indices):
                if cancel_event.is_set():
                    break

                t_q_start = time.perf_counter()
                try:
                    item = frame_queue.get(timeout=0.2)
                except queue.Empty:
                    if not worker_thread.is_alive():
                        if worker_error:
                            raise worker_error[0]
                        break
                    continue
                self._timings["queue_wait_sec"] += time.perf_counter() - t_q_start

                if item is None:
                    if worker_error:
                        raise worker_error[0]
                    break

                out_frame_idx, out_obj_ids, masks_bool = item
                abs_out_frame_idx = out_frame_idx + offset
                if abs_out_frame_idx >= end_frame_idx:
                    break

                expected_idx = expected_frame_indices[frames_yielded]
                if abs_out_frame_idx != expected_idx:
                    raise RuntimeError(
                        f"SAM2 propagation index mismatch: expected {expected_idx}, got {abs_out_frame_idx}"
                    )

                if len(out_obj_ids) == 0 or masks_bool is None:
                    yield abs_out_frame_idx, []
                    frames_yielded += 1
                    continue

                t_geom = time.perf_counter()
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

                self._timings["geometry_sec"] += time.perf_counter() - t_geom

                yield abs_out_frame_idx, frame_tracks
                frames_yielded += 1

            # Assert exact frame count: all expected frames must be yielded by the predictor
            if frames_yielded != len(expected_frame_indices):
                raise RuntimeError(
                    f"SAM2 propagation truncated: expected {len(expected_frame_indices)} frames "
                    f"[{start_frame_idx}, {end_frame_idx}), yielded only {frames_yielded}"
                )
        finally:
            cancel_event.set()
            # Drain queue so producer thread cannot remain blocked on frame_queue.put
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    break
            worker_thread.join(timeout=5.0)

    def propagate_stream(
        self,
        start_frame_idx: int,
        num_frames: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """Backwards compatibility adapter mapping to propagate(start, start + count)."""
        return self.propagate(
            start_frame_idx=start_frame_idx,
            end_frame_idx=start_frame_idx + num_frames,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    def propagate_window(
        self,
        start_frame_idx: int,
        num_frames: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Propagates and collects results into dictionary."""
        results: Dict[int, List[Dict[str, Any]]] = {}
        for f_idx, f_tracks in self.propagate(
            start_frame_idx=start_frame_idx,
            end_frame_idx=start_frame_idx + num_frames,
            scale_x=scale_x,
            scale_y=scale_y,
        ):
            results[f_idx] = f_tracks
        return results

    def retire_objects(self, obj_ids: List[int]):
        """
        Removes specified objects from active SAM 2 computation memory and adapter metadata.
        Prevents unbounded growth of active track tracking state. Surfaces removal failures.
        """
        if not self.initialized or self._inference_state is None:
            for oid in obj_ids:
                self._tracks_meta.pop(oid, None)
            return

        with torch.inference_mode(), _get_autocast_context(self.device, self.dtype, self.use_autocast):
            for oid in obj_ids:
                self._tracks_meta.pop(oid, None)
                if hasattr(self._predictor, "remove_object"):
                    try:
                        self._predictor.remove_object(
                            self._inference_state,
                            obj_id=oid,
                            strict=False,
                            need_output=False,
                        )
                    except Exception as rem_err:
                        print(f"[SAM2BenthicTracker] Warning removing object {oid} from inference state: {rem_err}")

    def reset_timings(self):
        """Resets timing accumulators for a new processing chunk."""
        for k in self._timings:
            self._timings[k] = 0.0

    def reset(self):
        """Resets active tracking state while keeping loaded model."""
        if self._predictor is not None and self._inference_state is not None:
            try:
                if hasattr(self._predictor, "reset_state"):
                    self._predictor.reset_state(self._inference_state)
            except Exception:
                pass
        self._inference_state = None
        self._tracks_meta.clear()
        self.initialized = False

    def close(self):
        """Releases all states and memory resources."""
        self.reset()
        self._predictor = None

    def get_timings(self) -> Dict[str, float]:
        """Returns timing breakdown across stages."""
        return {k: round(v, 4) for k, v in self._timings.items()}

    def get_track_metadata(self, track_id: int) -> Optional[Dict[str, Any]]:
        return self._tracks_meta.get(track_id)

    @property
    def total_tracks_count(self) -> int:
        return len(self._tracks_meta)
