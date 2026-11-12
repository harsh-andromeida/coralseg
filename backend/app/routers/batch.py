"""
Batch image analysis, queue management, and dataset export router for ReefVision.
"""

import os
import sys
import datetime
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
import numpy as np

from fastapi import APIRouter, HTTPException, Response

from core.models import check_models_download_status
from core.export import create_coco_dataset_zip, export_dataset_to_directory
from app.schemas import (
    BatchStartRequest,
    BatchPrioritizeRequest,
    ExportFolderRequest,
    ExportFolderResponse,
)
from app import state

logger = logging.getLogger("reefvision.batch")
router = APIRouter(tags=["Batch Processing & Dataset Export"])


def _check_models_status() -> tuple[bool, list]:
    """Helper that checks models download status, respecting test monkey patching on app.main."""
    try:
        import app.main as app_main
        checker = getattr(app_main, "check_models_download_status", check_models_download_status)
        return checker()
    except Exception:
        return check_models_download_status()


@router.post("/api/batch/start")
def start_batch_processing(req: BatchStartRequest):
    """Starts background bulk processing of multiple images."""
    all_ready, _ = _check_models_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are missing or deleted. Please download all models before starting batch processing.",
        )

    # Ensure all requested images are registered in memory
    dirs = state._resolve_demo_dirs()
    for img_name in req.images:
        if img_name not in state._IMAGE_STORE:
            found = False
            for sdir in dirs:
                target = os.path.join(sdir, img_name)
                if os.path.isfile(target):
                    state._IMAGE_STORE[img_name] = Image.open(target).convert("RGB")
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded in session")

    params = {
        "points_per_side": req.points_per_side,
        "iou_thresh": req.iou_thresh,
        "stability_thresh": req.stability_thresh,
        "min_area_px": req.min_area_px,
    }
    concurrency = req.concurrency if req.concurrency else state._DEFAULT_CONCURRENCY

    if not state.batch_processor:
        raise HTTPException(status_code=500, detail="BatchProcessor is not initialized.")

    return state.batch_processor.start_batch(
        images=req.images,
        params=params,
        concurrency=concurrency,
    )


@router.get("/api/batch/status")
def get_batch_status():
    """Polls real-time bulk processing progress across all images."""
    if not state.batch_processor:
        return {"is_running": False, "total": 0, "completed": 0, "failed": 0, "items": []}
    return state.batch_processor.get_status()


@router.post("/api/batch/pause")
def pause_batch_processing():
    """Pauses background batch processing."""
    if not state.batch_processor:
        return {"status": "paused"}
    return state.batch_processor.pause_batch()


@router.post("/api/batch/resume")
def resume_batch_processing():
    """Resumes background batch processing."""
    if not state.batch_processor:
        return {"status": "resumed"}
    return state.batch_processor.resume_batch()


@router.post("/api/batch/cancel")
def cancel_batch_processing():
    """Cancels remaining pending images in the active batch and unloads models from RAM."""
    if not state.batch_processor:
        return {"status": "cancelled"}
    return state.batch_processor.cancel_batch()


@router.post("/api/batch/prioritize")
def prioritize_batch_image(req: BatchPrioritizeRequest):
    """Bumps a specific image to the front of the background processing queue."""
    if not state.batch_processor:
        return {"prioritized": False, "image_name": req.image_name}
    prioritized = state.batch_processor.prioritize_image(req.image_name)
    return {"prioritized": prioritized, "image_name": req.image_name}


def _gather_processed_dataset_items(image_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Collects structured telemetry, taxonomy annotations, masks, and metadata
    for all images processed via batch processor, active analysis studio sessions, or caches.
    """
    # 1. Image names from batch processor
    processed_names = state.batch_processor.get_processed_images_list() if state.batch_processor else []

    # 2. Image names from _ENRICH_CACHE
    enrich_names = [
        name for name in state._IMAGE_STORE.keys()
        if any(k.startswith(f"{name}_") for k in state._ENRICH_CACHE.keys())
    ]

    # 3. Image names from _BASE_SEG_CACHE
    seg_names = [
        name for name in state._IMAGE_STORE.keys()
        if any(k.startswith(f"{name}_") for k in state._BASE_SEG_CACHE.keys())
    ]

    all_target_names = list(dict.fromkeys(processed_names + enrich_names + seg_names))

    # If explicit image_names filter is provided, restrict to them (or include if in store)
    if image_names:
        allowed = set(image_names)
        all_target_names = [n for n in all_target_names if n in allowed]
        for req_name in image_names:
            if req_name in state._IMAGE_STORE and req_name not in all_target_names:
                all_target_names.append(req_name)

    if not all_target_names:
        raise HTTPException(
            status_code=400,
            detail="No analyzed images available to export yet. Please analyze at least one image before exporting.",
        )

    processed_items = []
    for name in all_target_names:
        pil_img = state._IMAGE_STORE.get(name)
        w = pil_img.width if pil_img else 1024
        h = pil_img.height if pil_img else 768

        # Find raw enriched masks if present
        matching_enrich_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{name}_")]
        if matching_enrich_keys:
            masks_info, full_image_eval, health_summary = state._ENRICH_CACHE[matching_enrich_keys[-1]]
        else:
            matching_base_keys = [k for k in state._BASE_SEG_CACHE.keys() if k.startswith(f"{name}_")]
            if matching_base_keys:
                masks_info = state._BASE_SEG_CACHE[matching_base_keys[-1]]
            else:
                masks_info = []
            full_image_eval, health_summary = {}, {}

        batch_item = (state.batch_processor.get_image_result(name) or {}) if state.batch_processor else {}
        segments = list(batch_item.get("segments", []))
        if not segments and masks_info:
            # Reconstruct table segments
            for idx, m in enumerate(masks_info):
                tax = m.get("taxonomy", {})
                bl = m.get("bleaching", {})
                m_id = m.get("id", idx + 1)
                segments.append({
                    "id": m_id,
                    "id_str": f"#{m_id}",
                    "genus": tax.get("genus", "Coral"),
                    "growth_form": tax.get("growth_form", "-"),
                    "taxon_conf": round(float(tax.get("confidence_pct", 90.0)), 1),
                    "condition": "Bleached" if bl.get("is_bleached") else "Healthy",
                    "condition_conf": round(float(bl.get("confidence_pct", 90.0)), 1),
                    "area_pct": m.get("area_pct", 0.0),
                    "area_px": m.get("area_px", m.get("area", 0)),
                    "predicted_iou": round(float(m.get("predicted_iou", 0.90)), 3),
                    "color_hex": m.get("color_hex", "#00cccc"),
                    "centroid": [int(v) for v in m["centroid"]] if m.get("centroid") else None,
                    "bbox": [int(v) for v in m["bbox"]] if m.get("bbox") else None,
                })

        cov_pct = batch_item.get("coverage_pct", 0.0)
        if cov_pct == 0.0 and segments and w > 0 and h > 0:
            total_px = sum(s.get("area_px", 0) for s in segments)
            cov_pct = round((total_px / (w * h)) * 100.0, 2)

        bleach_pct = batch_item.get(
            "bleaching_prevalence_pct",
            health_summary.get("bleaching_prevalence_pct", 0.0),
        )

        processed_items.append({
            "image_name": name,
            "width": w,
            "height": h,
            "masks_info": masks_info,
            "segments": segments,
            "corals_count": len(segments),
            "coverage_pct": cov_pct,
            "bleaching_prevalence_pct": bleach_pct,
            "summary": batch_item.get("summary", {}),
            "health_summary": health_summary,
            "scene_eval": full_image_eval,
        })

    return processed_items


@router.get("/api/batch/export/default-path")
def get_default_export_path():
    """Returns a platform-appropriate default folder path for dataset export."""
    home = Path.home()
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"reefvision_dataset_{timestamp_str}"
    for candidate_name in ["Downloads", "Desktop", "Documents"]:
        candidate_dir = home / candidate_name
        if candidate_dir.exists() and candidate_dir.is_dir():
            return {
                "default_folder": str(candidate_dir / folder_name),
                "parent_dir": str(candidate_dir),
                "suggested_name": folder_name,
            }
    return {
        "default_folder": str(home / folder_name),
        "parent_dir": str(home),
        "suggested_name": folder_name,
    }


@router.get("/api/batch/export/coco-zip")
def export_batch_coco_dataset(image_name: Optional[str] = None):
    """
    Assembles and streams a complete structured dataset archive (.zip)
    containing COCO JSON annotations, original images, summary.csv, and dataset_summary.json
    for all images processed up to this point.
    """
    img_filter = [image_name] if image_name else None
    processed_items = _gather_processed_dataset_items(image_names=img_filter)

    zip_bytes = create_coco_dataset_zip(processed_items, state._IMAGE_STORE)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reefvision_coco_dataset_{timestamp_str}.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.post("/api/batch/export/to-folder", response_model=ExportFolderResponse)
def export_dataset_to_custom_folder(req: ExportFolderRequest):
    """
    Exports the complete structured dataset directly into a user-specified custom folder.
    Creates annotations/instances_default.json, images/, summary.csv, dataset_summary.json,
    and README.md directly in that folder, with an optional bundled .zip archive.
    """
    if not req.folder_path or not req.folder_path.strip():
        raise HTTPException(status_code=400, detail="Destination folder path cannot be empty.")

    target_path = Path(req.folder_path.strip()).expanduser().resolve()
    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot create or access destination folder: {e}")

    processed_items = _gather_processed_dataset_items(image_names=req.image_names)
    res = export_dataset_to_directory(
        processed_items=processed_items,
        image_store=state._IMAGE_STORE,
        output_dir=target_path,
        include_zip=req.include_zip,
    )

    if req.open_folder:
        try:
            if sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(target_path)])
            elif sys.platform == "win32":
                import os
                os.startfile(str(target_path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(target_path)])
        except Exception as e:
            logger.warning(f"Could not open folder in file manager: {e}")

    return ExportFolderResponse(
        success=True,
        folder_path=str(target_path),
        total_images=res.get("total_images", 0),
        images_saved=res.get("images_saved", 0),
        total_corals=res.get("total_corals", 0),
        written_files_count=res.get("written_files_count", 0),
        zip_file=res.get("zip_file"),
    )
