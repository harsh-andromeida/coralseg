"""
Autonomous Background Batch Processing Engine for ReefVision.
Handles asynchronous, multi-threaded instance segmentation, taxonomy enrichment,
overlay pre-rendering, priority queue management, and pause/resume/cancel lifecycles.
"""

import time
import queue
import threading
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
import numpy as np

from core.segmentation import run_segmentation
from core.taxonomy import enrich_masks_with_taxonomy_and_bleaching
from core.visualization import create_segmentation_overlay, generate_distinct_colors


class BatchCancelledException(Exception):
    """Raised when batch processing is cancelled cooperatively."""
    pass


class BatchProcessor:
    def __init__(
        self,
        image_store: Dict[str, Image.Image],
        base_seg_cache: Dict[str, List[Dict[str, Any]]],
        enrich_cache: Dict[str, Any],
        overlay_cache: Dict[str, str],
        model_loader_sam: Callable,
        model_loader_bioclip: Callable,
        model_loader_bleaching: Callable,
        get_active_device: Callable[[], str],
        model_unloader: Optional[Callable[[], Any]] = None,
        models_loaded_checker: Optional[Callable[[], Dict[str, bool]]] = None,
    ):
        self._image_store = image_store
        self._base_seg_cache = base_seg_cache
        self._enrich_cache = enrich_cache
        self._overlay_cache = overlay_cache
        self._model_loader_sam = model_loader_sam
        self._model_loader_bioclip = model_loader_bioclip
        self._model_loader_bleaching = model_loader_bleaching
        self._get_active_device = get_active_device
        self._model_unloader = model_unloader
        self._models_loaded_checker = models_loaded_checker

        self._lock = threading.Lock()
        self._batch_id = ""
        self._is_running = False
        self._is_paused = False
        self._cancel_requested = False
        self._concurrency = 1

        self._images_order: List[str] = []
        self._pending_list: List[str] = []
        self._active_workers: List[threading.Thread] = []

        self._items: Dict[str, Dict[str, Any]] = {}
        self._current_image: Optional[str] = None
        self._current_stage: str = "idle"
        self._completed_count = 0
        self._failed_count = 0

        # Hyperparameters for the run
        self._params = {
            "points_per_side": 16,
            "iou_thresh": 0.50,
            "stability_thresh": 0.50,
            "min_area_px": 100,
        }

    def _check_cancelled(self):
        with self._lock:
            if self._cancel_requested:
                raise BatchCancelledException("Batch cancelled by user")

    def start_batch(
        self,
        images: List[str],
        params: Optional[Dict[str, Any]] = None,
        concurrency: int = 1,
    ) -> Dict[str, Any]:
        with self._lock:
            # Stop existing batch if running
            self._cancel_requested = True
            self._is_running = False
            self._is_paused = False

            time.sleep(0.05)  # brief grace period for previous workers

            if params:
                self._params.update(params)

            self._batch_id = f"batch_{int(time.time() * 1000)}"
            self._concurrency = max(1, min(int(concurrency), 4))
            self._images_order = list(images)
            self._pending_list = list(images)
            self._cancel_requested = False
            self._is_running = True
            self._is_paused = False
            self._completed_count = 0
            self._failed_count = 0
            self._current_image = None
            self._current_stage = "initializing"

            self._items = {}
            for img_name in images:
                self._items[img_name] = {
                    "image_name": img_name,
                    "status": "pending",
                    "stage": "pending",
                    "error": None,
                    "corals_count": 0,
                    "coverage_pct": 0.0,
                    "bleaching_prevalence_pct": 0.0,
                    "summary": None,
                    "segments": None,
                    "health_summary": None,
                    "scene_eval": None,
                    "overlay_base64": None,
                }

            # Spawn worker threads
            self._active_workers = []
            for w_idx in range(self._concurrency):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"ReefVision-BatchWorker-{w_idx+1}",
                    daemon=True,
                )
                self._active_workers.append(t)
                t.start()

            return self.get_status_locked()

    def pause_batch(self) -> Dict[str, Any]:
        with self._lock:
            if self._is_running:
                self._is_paused = True
            return self.get_status_locked()

    def resume_batch(self) -> Dict[str, Any]:
        with self._lock:
            if self._is_running and self._is_paused:
                self._is_paused = False
            return self.get_status_locked()

    def cancel_batch(self) -> Dict[str, Any]:
        with self._lock:
            self._cancel_requested = True
            self._is_running = False
            self._is_paused = False

            # Immediately mark active and pending items as cancelled
            if self._current_image and self._current_image in self._items:
                if self._items[self._current_image]["status"] in ("pending", "processing"):
                    self._items[self._current_image]["status"] = "cancelled"
                    self._items[self._current_image]["stage"] = "cancelled"
            self._current_image = None

            for name in self._pending_list:
                if self._items[name]["status"] == "pending":
                    self._items[name]["status"] = "cancelled"
                    self._items[name]["stage"] = "cancelled"
            self._pending_list.clear()

        # Safely unload models from RAM and trigger garbage collection
        if self._model_unloader is not None:
            try:
                self._model_unloader()
            except Exception as err:
                print(f"[BatchProcessor] Error unloading models: {err}")

        with self._lock:
            self._current_stage = "Cancelled • Models unloaded from RAM"
            return self.get_status_locked()

    def prioritize_image(self, image_name: str) -> bool:
        with self._lock:
            if image_name in self._pending_list:
                self._pending_list.remove(image_name)
                self._pending_list.insert(0, image_name)
                return True
            return False

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self.get_status_locked()

    def get_status_locked(self) -> Dict[str, Any]:
        total = len(self._images_order)
        percent = int((self._completed_count / total * 100)) if total > 0 else 0
        return {
            "batch_id": self._batch_id,
            "is_running": self._is_running,
            "is_paused": self._is_paused,
            "concurrency": self._concurrency,
            "total": total,
            "completed": self._completed_count,
            "failed": self._failed_count,
            "percent": percent,
            "current_image": self._current_image,
            "current_stage": self._current_stage,
            "images_order": list(self._images_order),
            "items": {k: dict(v) for k, v in self._items.items()},
        }

    def get_processed_images_list(self) -> List[str]:
        with self._lock:
            return [
                name for name in self._images_order
                if self._items.get(name, {}).get("status") == "completed"
            ]

    def get_image_result(self, image_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.get(image_name)
            if not item:
                return None
            return dict(item)

    def _worker_loop(self):
        while True:
            # Check for cancellation or finish
            with self._lock:
                if self._cancel_requested:
                    break

                # Handle pause
                if self._is_paused:
                    pass  # release lock and wait below

                if not self._pending_list:
                    # Check if all done
                    all_done = all(
                        v["status"] in ("completed", "error", "cancelled")
                        for v in self._items.values()
                    )
                    if all_done:
                        self._is_running = False
                        self._current_stage = "completed"
                        self._current_image = None
                    break

            if self._is_paused:
                time.sleep(0.3)
                continue

            # Pick next image
            target_image: Optional[str] = None
            with self._lock:
                if self._pending_list and not self._cancel_requested and not self._is_paused:
                    target_image = self._pending_list.pop(0)
                    self._items[target_image]["status"] = "processing"
                    self._items[target_image]["stage"] = "segmenting"
                    self._current_image = target_image
                    self._current_stage = "segmenting"

            if not target_image:
                time.sleep(0.05)
                continue

            # Process the image with cooperative cancellation
            try:
                self._process_single_image(target_image)
                with self._lock:
                    if not self._cancel_requested:
                        self._items[target_image]["status"] = "completed"
                        self._items[target_image]["stage"] = "ready"
                        self._completed_count += 1
            except BatchCancelledException:
                with self._lock:
                    self._items[target_image]["status"] = "cancelled"
                    self._items[target_image]["stage"] = "cancelled"
                    self._current_image = None
                break
            except Exception as e:
                with self._lock:
                    self._items[target_image]["status"] = "error"
                    self._items[target_image]["stage"] = "error"
                    self._items[target_image]["error"] = str(e)
                    self._failed_count += 1

            # Update overall stage
            with self._lock:
                if not self._pending_list:
                    all_done = all(
                        v["status"] in ("completed", "error", "cancelled")
                        for v in self._items.values()
                    )
                    if all_done:
                        self._is_running = False
                        if not self._cancel_requested:
                            self._current_stage = "completed"
                        self._current_image = None

    def _process_single_image(self, img_name: str):
        self._check_cancelled()

        if img_name not in self._image_store:
            raise ValueError(f"Image '{img_name}' is not registered in session image store")

        pil_img = self._image_store[img_name]
        img_w, img_h = pil_img.width, pil_img.height
        total_pixels = img_w * img_h
        active_device = self._get_active_device()

        pts = self._params["points_per_side"]
        iou_t = self._params["iou_thresh"]
        stab_t = self._params["stability_thresh"]
        min_area = self._params["min_area_px"]

        base_cache_key = f"{img_name}_{pts}_{iou_t}_{stab_t}"

        # Step 0: Ensure SAM model is loaded in RAM (show loading stage in UI if not)
        if base_cache_key not in self._base_seg_cache:
            is_sam_in_ram = (
                self._models_loaded_checker().get("sam_loaded", False)
                if self._models_loaded_checker
                else False
            )
            if not is_sam_in_ram:
                with self._lock:
                    self._items[img_name]["stage"] = "loading_sam"
                    self._current_stage = f"Loading SAM ViT-B model into RAM ({img_name})..."
            self._check_cancelled()
            sam_model = self._model_loader_sam(device=active_device)
            self._check_cancelled()

            # Step 1: SAM ViT-B Dense Segmentation
            with self._lock:
                self._items[img_name]["stage"] = "segmenting"
                self._current_stage = f"Segmenting {img_name} (SAM ViT-B)"

            img_np = np.array(pil_img)
            masks_info, _ = run_segmentation(
                model=sam_model,
                image=img_np,
                points_per_side=pts,
                pred_iou_thresh=iou_t,
                stability_score_thresh=stab_t,
                min_mask_region_area=0,
            )
            self._base_seg_cache[base_cache_key] = masks_info

        self._check_cancelled()
        all_candidate_masks = self._base_seg_cache[base_cache_key]

        # Filter masks by min_area_px
        raw_filtered = [
            m for m in all_candidate_masks
            if m.get("area_px", m.get("area", 0)) >= min_area
        ]

        distinct_colors = generate_distinct_colors(len(raw_filtered))
        filtered_masks = []
        for new_idx, m in enumerate(raw_filtered):
            m_copy = dict(m)
            m_id = new_idx + 1
            m_copy["id"] = m_id
            m_copy["segment_id"] = m_id
            m_area = m_copy.get("area_px", m_copy.get("area", 0))
            m_copy["area"] = m_area
            m_copy["area_px"] = m_area
            m_copy["area_pct"] = round((m_area / max(total_pixels, 1)) * 100.0, 2)
            c_rgb = distinct_colors[new_idx]
            m_copy["color_rgb"] = c_rgb
            m_copy["instance_color_rgb"] = c_rgb
            m_copy["color_hex"] = f"#{c_rgb[0]:02x}{c_rgb[1]:02x}{c_rgb[2]:02x}"
            m_copy["instance_color_hex"] = m_copy["color_hex"]
            filtered_masks.append(m_copy)

        self._check_cancelled()

        # Step 2: BioCLIP Taxonomy & Bleaching Enrichment
        enrich_cache_key = f"{base_cache_key}_{min_area}"
        if enrich_cache_key not in self._enrich_cache:
            is_bioclip_in_ram = (
                self._models_loaded_checker().get("bioclip_loaded", False)
                if self._models_loaded_checker
                else False
            )
            if not is_bioclip_in_ram:
                with self._lock:
                    self._items[img_name]["stage"] = "loading_bioclip"
                    self._current_stage = f"Loading BioCLIP & YOLO11 models into RAM ({img_name})..."
            self._check_cancelled()
            bioclip = self._model_loader_bioclip(device=active_device)
            bleaching = self._model_loader_bleaching(device=active_device)
            self._check_cancelled()

            with self._lock:
                self._items[img_name]["stage"] = "classifying"
                self._current_stage = f"Classifying {img_name} (BioCLIP & YOLO11)"

            enriched_masks, full_image_eval, health_summary = enrich_masks_with_taxonomy_and_bleaching(
                image=pil_img,
                masks_info=filtered_masks,
                bioclip_bundle=bioclip,
                bleaching_model=bleaching,
            )
            self._enrich_cache[enrich_cache_key] = (enriched_masks, full_image_eval, health_summary)

        self._check_cancelled()
        masks_info, full_image_eval, health_summary = self._enrich_cache[enrich_cache_key]

        # Recompute summary stats
        union_mask = np.zeros((img_h, img_w), dtype=bool)
        for m in masks_info:
            seg = m.get("mask", m.get("segmentation"))
            if seg is not None:
                union_mask = np.logical_or(union_mask, seg)

        coral_covered_pixels = int(np.sum(union_mask))
        coral_coverage_pct = round((coral_covered_pixels / max(total_pixels, 1)) * 100.0, 2)
        mean_iou = round(float(np.mean([m["predicted_iou"] for m in masks_info])), 4) if masks_info else 0.0
        mean_stability = round(float(np.mean([m["stability_score"] for m in masks_info])), 4) if masks_info else 0.0

        summary_stats = {
            "total_corals_detected": len(masks_info),
            "coral_coverage_pct": coral_coverage_pct,
            "coral_covered_pixels": coral_covered_pixels,
            "total_image_pixels": total_pixels,
            "image_resolution": f"{img_w}x{img_h}",
            "mean_iou_confidence": mean_iou,
            "mean_stability_score": mean_stability,
        }

        # Serializable table segments
        table_records = []
        for m in masks_info:
            tax = m.get("taxonomy", {})
            bl = m.get("bleaching", {})
            cond_label = "Bleached" if bl.get("is_bleached") else "Healthy"
            table_records.append({
                "id": m["id"],
                "id_str": f"#{m['id']}",
                "genus": tax.get("genus", "Coral"),
                "growth_form": tax.get("growth_form", "-"),
                "taxon_conf": round(float(tax.get("confidence_pct", 0)), 1),
                "condition": cond_label,
                "condition_conf": round(float(bl.get("confidence_pct", 0)), 1),
                "area_pct": m.get("area_pct", 0.0),
                "area_px": m.get("area_px", m.get("area", 0)),
                "predicted_iou": round(float(m["predicted_iou"]), 3),
                "color_hex": m.get("color_hex", "#00cccc"),
                "centroid": [int(v) for v in m["centroid"]] if m.get("centroid") else None,
                "bbox": [int(v) for v in m["bbox"]] if m.get("bbox") else None,
            })

        self._check_cancelled()

        # Step 3: Pre-render default overlay PNG base64
        with self._lock:
            self._items[img_name]["stage"] = "rendering"
            self._current_stage = f"Rendering Overlay for {img_name}"

        import io
        import base64
        overlay_res = create_segmentation_overlay(
            image=pil_img,
            masks_info=masks_info,
            alpha=0.45,
            draw_contours=True,
            draw_labels=True,
            draw_boxes=False,
            selected_mask_id=None,
            color_mode="instance",
        )
        overlay_pil = overlay_res if isinstance(overlay_res, Image.Image) else Image.fromarray(overlay_res)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)
        overlay_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        overlay_cache_key = f"{img_name}_default"
        self._overlay_cache[overlay_cache_key] = overlay_b64

        self._check_cancelled()

        # Commit results to item dictionary
        with self._lock:
            item = self._items[img_name]
            item["corals_count"] = len(table_records)
            item["coverage_pct"] = coral_coverage_pct
            item["bleaching_prevalence_pct"] = health_summary.get("bleaching_prevalence_pct", 0.0)
            item["summary"] = summary_stats
            item["segments"] = table_records
            item["health_summary"] = health_summary
            item["scene_eval"] = full_image_eval
            item["overlay_base64"] = overlay_b64
