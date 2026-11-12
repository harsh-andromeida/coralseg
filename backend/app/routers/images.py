"""
Image analysis, SAM segmentation, manual annotation, and export router for ReefVision.
"""

import os
import io
import glob
import base64
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse

from core.models import (
    check_models_download_status,
    load_coralscop_model,
    load_bioclip_model,
    load_bleaching_model,
    get_cached_model_path,
    HF_SAM_ENCODER_FILE,
    HF_SAM_DECODER_FILE,
)
from core.segmentation import run_segmentation
from core.taxonomy import enrich_masks_with_taxonomy_and_bleaching
from core.visualization import create_segmentation_overlay, generate_distinct_colors
from core.export import build_coco_json
from core.interactive_segmentation import (
    SAM_SESSION,
    mask_to_polygon,
    polygon_to_mask,
    calculate_polygon_metrics,
)
from app.schemas import (
    SegmentRequest,
    EnrichRequest,
    OverlayRequest,
    ExportCocoRequest,
    SegmentPromptRequest,
    SegmentPromptWarmupRequest,
    SaveAnnotationsRequest,
)
from app import state

router = APIRouter(tags=["Images & Single Image Analysis"])


def _check_models_status() -> tuple[bool, list]:
    """Helper that checks models download status, respecting test monkey patching on app.main."""
    try:
        import app.main as app_main
        checker = getattr(app_main, "check_models_download_status", check_models_download_status)
        return checker()
    except Exception:
        return check_models_download_status()


@router.get("/api/samples")
def list_samples() -> List[Dict[str, Any]]:
    """Discovers available coral demo samples."""
    dirs = state._resolve_demo_dirs()
    found = []
    seen = set()
    for sdir in dirs:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for fp in sorted(glob.glob(os.path.join(sdir, ext))):
                fn = os.path.basename(fp)
                if fn not in seen:
                    seen.add(fn)
                    found.append({
                        "filename": fn,
                        "path": fp,
                        "size_bytes": os.path.getsize(fp),
                    })
    return found


@router.get("/api/samples/image/{filename}")
def get_sample_image(filename: str):
    """Streams a sample image file."""
    dirs = state._resolve_demo_dirs()
    for sdir in dirs:
        target = os.path.join(sdir, filename)
        if os.path.isfile(target):
            with open(target, "rb") as f:
                content = f.read()
            ext = filename.rsplit(".", 1)[-1].lower()
            media_type = "image/png" if ext == "png" else "image/jpeg"
            return Response(content=content, media_type=media_type)
    raise HTTPException(status_code=404, detail=f"Sample '{filename}' not found")


@router.post("/api/images/register")
def register_image_base64(payload: Dict[str, str]):
    """Registers an image in memory via base64."""
    image_name = payload.get("filename", f"upload_{int(time.time())}.png")
    data_str = payload.get("data", "")
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    raw_bytes = base64.b64decode(data_str)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    state._IMAGE_STORE[image_name] = pil_img
    return {
        "status": "registered",
        "filename": image_name,
        "width": pil_img.width,
        "height": pil_img.height,
    }


@router.post("/api/images/load-sample")
def load_sample_to_store(payload: Dict[str, str]):
    """Loads a named sample image from disk into memory."""
    filename = payload.get("filename", "")
    dirs = state._resolve_demo_dirs()
    for sdir in dirs:
        target = os.path.join(sdir, filename)
        if os.path.isfile(target):
            pil_img = Image.open(target).convert("RGB")
            state._IMAGE_STORE[filename] = pil_img
            return {
                "status": "loaded",
                "filename": filename,
                "width": pil_img.width,
                "height": pil_img.height,
            }
    raise HTTPException(status_code=404, detail=f"Sample '{filename}' not found")


@router.post("/api/analysis/segment")
def segment_image(req: SegmentRequest):
    """Runs CoralSCOP SAM segmentation on a registered image."""
    all_ready, _ = _check_models_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are not downloaded or have been deleted. You must download all models before running segmentation.",
        )

    img_name = req.image_name
    if img_name not in state._IMAGE_STORE:
        # If image_base64 is provided in the request, register it automatically
        if req.image_base64:
            data_str = req.image_base64
            if "," in data_str:
                data_str = data_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(data_str)
            state._IMAGE_STORE[img_name] = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        else:
            # Check if it exists in demo_images
            dirs = state._resolve_demo_dirs()
            found = False
            for sdir in dirs:
                target = os.path.join(sdir, img_name)
                if os.path.isfile(target):
                    state._IMAGE_STORE[img_name] = Image.open(target).convert("RGB")
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded in session")

    pil_img = state._IMAGE_STORE[img_name]
    cache_key = f"{img_name}_{req.points_per_side}_{req.iou_thresh}_{req.stability_thresh}"

    if cache_key not in state._BASE_SEG_CACHE:
        seg_lock = state._get_seg_lock(cache_key)
        with seg_lock:
            if cache_key not in state._BASE_SEG_CACHE:
                sam_bundle = load_coralscop_model(device=state._ACTIVE_DEVICE_PREF)
                img_np = np.array(pil_img)
                masks_info, _ = run_segmentation(
                    model=sam_bundle,
                    image=img_np,
                    points_per_side=req.points_per_side,
                    pred_iou_thresh=req.iou_thresh,
                    stability_score_thresh=req.stability_thresh,
                    min_mask_region_area=0,
                    image_name=img_name,
                )
                state._BASE_SEG_CACHE[cache_key] = masks_info

    all_candidate_masks = state._BASE_SEG_CACHE[cache_key]
    return {
        "status": "success",
        "cache_key": cache_key,
        "total_candidate_masks": len(all_candidate_masks),
        "image_width": pil_img.width,
        "image_height": pil_img.height,
    }


@router.post("/api/analysis/enrich")
def enrich_image(req: EnrichRequest):
    """Enriches filtered masks with BioCLIP taxonomy and NOAA bleaching detection."""
    all_ready, _ = _check_models_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are not downloaded or have been deleted. You must download all models before running taxonomy enrichment.",
        )

    img_name = req.image_name
    if img_name not in state._IMAGE_STORE:
        raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded")

    pil_img = state._IMAGE_STORE[img_name]
    img_h, img_w = pil_img.height, pil_img.width
    total_pixels = img_h * img_w

    # Find the most recent base seg cache key for this image
    matching_keys = [k for k in state._BASE_SEG_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /segment first")
    base_cache_key = matching_keys[-1]
    all_candidate_masks = state._BASE_SEG_CACHE[base_cache_key]

    # Filter masks by min_area_px
    raw_filtered = [
        m for m in all_candidate_masks
        if m.get("area_px", m.get("area", 0)) >= req.min_area_px
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

    enrich_cache_key = f"{base_cache_key}_{req.min_area_px}"
    if enrich_cache_key not in state._ENRICH_CACHE:
        bioclip = load_bioclip_model(device=state._ACTIVE_DEVICE_PREF)
        bleaching = load_bleaching_model(device=state._ACTIVE_DEVICE_PREF)

        enriched_masks, full_image_eval, health_summary = enrich_masks_with_taxonomy_and_bleaching(
            image=pil_img,
            masks_info=filtered_masks,
            bioclip_bundle=bioclip,
            bleaching_model=bleaching,
        )
        state._ENRICH_CACHE[enrich_cache_key] = (enriched_masks, full_image_eval, health_summary)

    masks_info, full_image_eval, health_summary = state._ENRICH_CACHE[enrich_cache_key]

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

    # Prepare lightweight serializable list for table
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

    return {
        "summary": summary_stats,
        "segments": table_records,
        "health_summary": health_summary,
        "scene_eval": full_image_eval,
    }


@router.post("/api/analysis/overlay")
def render_overlay(req: OverlayRequest):
    """Renders high-contrast segmentation overlay on image canvas."""
    img_name = req.image_name
    if img_name not in state._IMAGE_STORE:
        raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded")

    pil_img = state._IMAGE_STORE[img_name]

    # Find enriched masks
    matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{img_name}_") and k.endswith(f"_{req.min_area_px}")]
    if not matching_keys:
        # Fallback to any enriched cache for this image
        matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /enrich first")

    enrich_cache_key = matching_keys[-1]
    masks_info, _, _ = state._ENRICH_CACHE[enrich_cache_key]

    base_image = pil_img
    if req.layout_mode == "Masks on Black":
        base_image = np.zeros_like(np.array(pil_img))

    overlay_res = create_segmentation_overlay(
        image=base_image,
        masks_info=masks_info,
        alpha=req.alpha,
        draw_contours=req.draw_contours,
        draw_labels=req.draw_labels,
        draw_boxes=req.draw_boxes,
        selected_mask_id=req.selected_mask_id,
        color_mode=req.color_mode,
    )

    overlay_pil = overlay_res if isinstance(overlay_res, Image.Image) else Image.fromarray(overlay_res)

    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    buf.seek(0)
    base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "overlay_base64": f"data:image/png;base64,{base64_str}",
        "width": overlay_pil.width,
        "height": overlay_pil.height,
    }


@router.get("/api/analysis/polygons/{image_name}")
def get_image_polygons(image_name: str):
    """
    Returns polygon boundaries and metadata for all detected or human-annotated
    coral instances for an image.
    """
    if image_name not in state._IMAGE_STORE:
        # Try finding in demo_images
        dirs = state._resolve_demo_dirs()
        for sdir in dirs:
            target = os.path.join(sdir, image_name)
            if os.path.isfile(target):
                state._IMAGE_STORE[image_name] = Image.open(target).convert("RGB")
                break
        if image_name not in state._IMAGE_STORE:
            raise HTTPException(status_code=404, detail=f"Image '{image_name}' not loaded in session")

    pil_img = state._IMAGE_STORE[image_name]
    img_h, img_w = pil_img.height, pil_img.width

    matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{image_name}_")]
    masks_info = []
    if matching_keys:
        masks_info, _, _ = state._ENRICH_CACHE[matching_keys[-1]]
    else:
        # Check base seg cache
        base_keys = [k for k in state._BASE_SEG_CACHE.keys() if k.startswith(f"{image_name}_")]
        if base_keys:
            masks_info = state._BASE_SEG_CACHE[base_keys[-1]]

    results = []
    for m in masks_info:
        seg_id = m.get("id", m.get("segment_id", len(results) + 1))
        tax = m.get("taxonomy", {})
        bl = m.get("bleaching", {})

        # Use existing polygon if present, else trace outer contour
        poly = m.get("polygon")
        if poly is None:
            seg_mask = m.get("mask", m.get("segmentation"))
            if seg_mask is not None:
                poly = mask_to_polygon(seg_mask, tolerance=1.8, max_points=100)
                m["polygon"] = poly
            else:
                poly = []

        is_human = m.get("is_human_annotated", False)
        taxon_conf = 100.0 if is_human else round(float(tax.get("confidence_pct", 0.0)), 1)
        cond_conf = 100.0 if is_human else round(float(bl.get("confidence_pct", 0.0)), 1)
        iou_val = 1.0 if is_human else round(float(m.get("predicted_iou", 0.0)), 3)

        cond_label = "Bleached" if bl.get("is_bleached") else "Healthy"

        results.append({
            "id": seg_id,
            "id_str": f"#{seg_id}",
            "genus": tax.get("genus", "Coral"),
            "growth_form": tax.get("growth_form", "-"),
            "taxon_conf": taxon_conf,
            "condition": cond_label,
            "condition_conf": cond_conf,
            "area_pct": m.get("area_pct", 0.0),
            "area_px": m.get("area_px", m.get("area", 0)),
            "predicted_iou": iou_val,
            "color_hex": m.get("color_hex", "#00cccc"),
            "polygon": poly,
            "bbox": [int(v) for v in m["bbox"]] if m.get("bbox") else None,
            "centroid": [int(v) for v in m["centroid"]] if m.get("centroid") else None,
            "is_human_annotated": is_human,
        })

    return {
        "image_name": image_name,
        "width": img_w,
        "height": img_h,
        "segments": results,
    }


def _resolve_session_image(img_name: str, image_data: Optional[str] = None) -> Optional[Image.Image]:
    """Helper to retrieve or decode an image into state._IMAGE_STORE."""
    if img_name in state._IMAGE_STORE:
        return state._IMAGE_STORE[img_name]

    if image_data:
        data_str = image_data
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(data_str)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            state._IMAGE_STORE[img_name] = pil_img
            return pil_img
        except Exception:
            pass

    # Check demo images and unlabeled samples
    dirs = state._resolve_demo_dirs()
    for sdir in dirs:
        target = os.path.join(sdir, img_name)
        if os.path.isfile(target):
            try:
                pil_img = Image.open(target).convert("RGB")
                state._IMAGE_STORE[img_name] = pil_img
                return pil_img
            except Exception:
                pass
    return None


@router.post("/api/analysis/segment-prompt")
def run_interactive_prompt_segmentation(req: SegmentPromptRequest):
    """
    Executes interactive SAM point prompt segmentation.
    Given click coordinates and positive/negative labels, generates a crisp mask polygon in ~15-25ms.
    """
    sam_enc = get_cached_model_path(HF_SAM_ENCODER_FILE)
    sam_dec = get_cached_model_path(HF_SAM_DECODER_FILE)
    if not sam_enc or not sam_dec:
        raise HTTPException(
            status_code=428,
            detail="SAM segmentation foundation models (encoder/decoder) are not downloaded. Please download models first.",
        )

    img_name = req.image_name
    pil_img = _resolve_session_image(img_name, req.image_data)
    if pil_img is None:
        raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded in session")

    sam_bundle = load_coralscop_model(device=state._ACTIVE_DEVICE_PREF)

    pts = [{"x": p.x, "y": p.y, "label": p.label} for p in req.points]
    try:
        result = SAM_SESSION.prompt_segment(
            sam_bundle=sam_bundle,
            image_name=img_name,
            pil_image=pil_img,
            points=pts,
        )
        return {
            "status": "success",
            "image_name": img_name,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interactive segmentation failed: {str(e)}")


@router.post("/api/analysis/segment-prompt/warmup")
def warmup_interactive_sam(req: SegmentPromptWarmupRequest):
    """
    Precomputes SAM image embeddings for an image ahead of time,
    so that subsequent prompt segment queries execute in ~15-30ms without delay.
    """
    sam_enc = get_cached_model_path(HF_SAM_ENCODER_FILE)
    sam_dec = get_cached_model_path(HF_SAM_DECODER_FILE)
    if not sam_enc or not sam_dec:
        return {"status": "skipped", "reason": "models_not_downloaded"}

    img_name = req.image_name
    pil_img = _resolve_session_image(img_name, req.image_data)
    if pil_img is None:
        return {"status": "skipped", "reason": "image_not_found"}

    sam_bundle = load_coralscop_model(device=state._ACTIVE_DEVICE_PREF)
    SAM_SESSION.get_or_compute_embeddings(sam_bundle, img_name, pil_img)
    return {"status": "warmed_up", "image_name": img_name}


@router.post("/api/analysis/save-annotations")
def save_manual_annotations(req: SaveAnnotationsRequest):
    """
    Saves user-edited and newly created coral instance annotations for an image.
    Marks all modified annotations as 100% confidence human ground truth.
    Recalculates exact area, percent cover, health metrics, and re-renders overlay.
    """
    img_name = req.image_name
    if img_name not in state._IMAGE_STORE:
        if req.image_data and req.image_data.startswith("data:"):
            header, b64_str = req.image_data.split(",", 1)
            raw_bytes = base64.b64decode(b64_str)
            state._IMAGE_STORE[img_name] = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        else:
            # Check demo images
            dirs = state._resolve_demo_dirs()
            for sdir in dirs:
                target = os.path.join(sdir, img_name)
                if os.path.isfile(target):
                    state._IMAGE_STORE[img_name] = Image.open(target).convert("RGB")
                    break
        if img_name not in state._IMAGE_STORE:
            raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded in session")

    pil_img = state._IMAGE_STORE[img_name]
    img_h, img_w = pil_img.height, pil_img.width
    total_pixels = img_h * img_w

    masks_info = []
    union_mask = np.zeros((img_h, img_w), dtype=bool)

    distinct_colors = generate_distinct_colors(max(len(req.segments), 1))

    for idx, item in enumerate(req.segments):
        m_id = item.id if item.id > 0 else (idx + 1)
        c_rgb = distinct_colors[idx % len(distinct_colors)]
        c_hex = item.color_hex or f"#{c_rgb[0]:02x}{c_rgb[1]:02x}{c_rgb[2]:02x}"

        # Parse RGB if hex provided
        if item.color_hex and item.color_hex.startswith("#") and len(item.color_hex) == 7:
            try:
                c_rgb = (
                    int(item.color_hex[1:3], 16),
                    int(item.color_hex[3:5], 16),
                    int(item.color_hex[5:7], 16),
                )
            except Exception:
                pass

        mask = polygon_to_mask(item.polygon, img_h, img_w)
        metrics = calculate_polygon_metrics(item.polygon, img_h, img_w)
        union_mask = np.logical_or(union_mask, mask)

        is_bleached = item.condition.strip().lower() == "bleached"

        mask_dict = {
            "id": m_id,
            "segment_id": m_id,
            "mask": mask,
            "segmentation": mask,
            "polygon": item.polygon,
            "area": metrics["area_px"],
            "area_px": metrics["area_px"],
            "area_pct": metrics["area_pct"],
            "bbox": metrics["bbox"],
            "centroid": metrics["centroid"],
            "predicted_iou": 1.0,
            "stability_score": 1.0,
            "color_rgb": c_rgb,
            "instance_color_rgb": c_rgb,
            "color_hex": c_hex,
            "instance_color_hex": c_hex,
            "is_human_annotated": True,
            "taxonomy": {
                "genus": item.genus or "Coral",
                "growth_form": item.growth_form or "-",
                "confidence_pct": 100.0,
                "primary_label": item.genus or "Coral",
            },
            "bleaching": {
                "is_bleached": is_bleached,
                "status": "Bleached" if is_bleached else "Healthy",
                "confidence_pct": 100.0,
                "bleached_prob": 1.0 if is_bleached else 0.0,
            },
        }
        masks_info.append(mask_dict)

    # Recalculate summary metrics
    coral_covered_pixels = int(np.sum(union_mask))
    coral_coverage_pct = round((coral_covered_pixels / max(total_pixels, 1)) * 100.0, 2)

    summary_stats = {
        "total_corals_detected": len(masks_info),
        "coral_coverage_pct": coral_coverage_pct,
        "coral_covered_pixels": coral_covered_pixels,
        "total_image_pixels": total_pixels,
        "image_resolution": f"{img_w}x{img_h}",
        "mean_iou_confidence": 1.0 if masks_info else 0.0,
        "mean_stability_score": 1.0 if masks_info else 0.0,
    }

    bleached_count = sum(1 for m in masks_info if m["bleaching"]["is_bleached"])
    healthy_count = len(masks_info) - bleached_count
    bleaching_prevalence_pct = round((bleached_count / max(len(masks_info), 1)) * 100.0, 1)

    health_summary = {
        "healthy_count": healthy_count,
        "bleached_count": bleached_count,
        "bleaching_prevalence_pct": bleaching_prevalence_pct,
    }

    # Lightweight table records
    table_records = []
    for m in masks_info:
        tax = m["taxonomy"]
        bl = m["bleaching"]
        table_records.append({
            "id": m["id"],
            "id_str": f"#{m['id']}",
            "genus": tax.get("genus", "Coral"),
            "growth_form": tax.get("growth_form", "-"),
            "taxon_conf": 100.0,
            "condition": "Bleached" if bl["is_bleached"] else "Healthy",
            "condition_conf": 100.0,
            "area_pct": m["area_pct"],
            "area_px": m["area_px"],
            "predicted_iou": 1.0,
            "color_hex": m["color_hex"],
            "centroid": list(m["centroid"]) if m.get("centroid") else None,
            "bbox": list(m["bbox"]) if m.get("bbox") else None,
            "is_human_annotated": True,
        })

    # Update in-memory inference caches
    cache_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not cache_keys:
        cache_keys = [f"{img_name}_16_0.5_0.5_100"]
    for k in cache_keys:
        state._ENRICH_CACHE[k] = (masks_info, {}, health_summary)

    base_cache_keys = [k for k in state._BASE_SEG_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not base_cache_keys:
        base_cache_keys = [f"{img_name}_16_0.5_0.5"]
    for bk in base_cache_keys:
        state._BASE_SEG_CACHE[bk] = masks_info

    # Pre-render fresh overlay
    overlay_res = create_segmentation_overlay(
        image=pil_img,
        masks_info=masks_info,
        alpha=0.45,
        draw_contours=True,
        draw_labels=True,
        draw_boxes=False,
        color_mode="instance",
    )
    overlay_pil = overlay_res if isinstance(overlay_res, Image.Image) else Image.fromarray(overlay_res)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    buf.seek(0)
    base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    overlay_data_url = f"data:image/png;base64,{base64_str}"
    state._OVERLAY_CACHE[img_name] = overlay_data_url

    # Update batch processor results if present
    if state.batch_processor and hasattr(state.batch_processor, "_results") and img_name in state.batch_processor._results:
        state.batch_processor._results[img_name].update({
            "status": "completed",
            "corals_count": len(masks_info),
            "coverage_pct": coral_coverage_pct,
            "bleaching_prevalence_pct": bleaching_prevalence_pct,
            "summary": summary_stats,
            "segments": table_records,
            "health_summary": health_summary,
            "overlay_base64": overlay_data_url,
        })

    return {
        "status": "success",
        "image_name": img_name,
        "summary": summary_stats,
        "segments": table_records,
        "health_summary": health_summary,
        "overlay_base64": overlay_data_url,
    }


@router.get("/api/analysis/result/{image_name}")
def get_analysis_result(image_name: str):
    """
    Returns precomputed results (segments, stats, health, and overlay)
    for a completed image, or its current background queue state.
    """
    # 1. Check BatchProcessor items
    item = state.batch_processor.get_image_result(image_name) if state.batch_processor else None
    if item and item.get("status") == "completed" and item.get("segments"):
        return {
            "status": "completed",
            "image_name": image_name,
            "corals_count": item.get("corals_count", 0),
            "coverage_pct": item.get("coverage_pct", 0.0),
            "bleaching_prevalence_pct": item.get("bleaching_prevalence_pct", 0.0),
            "summary": item.get("summary"),
            "segments": item.get("segments"),
            "health_summary": item.get("health_summary"),
            "scene_eval": item.get("scene_eval"),
            "overlay_base64": item.get("overlay_base64"),
        }

    # 2. Check in-memory _ENRICH_CACHE directly if processed outside batch
    matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{image_name}_")]
    if matching_keys:
        enrich_cache_key = matching_keys[-1]
        masks_info, full_image_eval, health_summary = state._ENRICH_CACHE[enrich_cache_key]
        pil_img = state._IMAGE_STORE.get(image_name)
        img_w = pil_img.width if pil_img else 1024
        img_h = pil_img.height if pil_img else 768
        total_pixels = img_w * img_h

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

        overlay_b64 = state._OVERLAY_CACHE.get(f"{image_name}_default")

        return {
            "status": "completed",
            "image_name": image_name,
            "corals_count": len(table_records),
            "coverage_pct": coral_coverage_pct,
            "bleaching_prevalence_pct": health_summary.get("bleaching_prevalence_pct", 0.0),
            "summary": summary_stats,
            "segments": table_records,
            "health_summary": health_summary,
            "scene_eval": full_image_eval,
            "overlay_base64": overlay_b64,
        }

    # 3. Otherwise return current pending/processing state
    return {
        "status": item.get("status", "pending") if item else "pending",
        "stage": item.get("stage", "pending") if item else "pending",
        "image_name": image_name,
        "error": item.get("error") if item else None,
    }


@router.post("/api/analysis/export/coco")
def export_coco(req: ExportCocoRequest):
    """Builds and returns standard COCO JSON format."""
    img_name = req.image_name
    if img_name not in state._IMAGE_STORE:
        raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded")

    pil_img = state._IMAGE_STORE[img_name]
    matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{img_name}_") and k.endswith(f"_{req.min_area_px}")]
    if not matching_keys:
        matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /enrich first")

    enrich_cache_key = matching_keys[-1]
    masks_info, full_image_eval, health_summary = state._ENRICH_CACHE[enrich_cache_key]

    coco_dict = build_coco_json(
        image_name=img_name,
        width=pil_img.width,
        height=pil_img.height,
        masks_info=masks_info,
    )
    coco_dict["health_summary"] = health_summary
    coco_dict["full_image_taxonomy"] = full_image_eval.get("taxonomy")
    coco_dict["full_image_bleaching"] = full_image_eval.get("bleaching")

    return JSONResponse(content=coco_dict)


@router.post("/api/analysis/export/csv")
def export_csv(req: ExportCocoRequest):
    """Exports segments table as CSV text."""
    img_name = req.image_name
    matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{img_name}_") and k.endswith(f"_{req.min_area_px}")]
    if not matching_keys:
        matching_keys = [k for k in state._ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /enrich first")

    enrich_cache_key = matching_keys[-1]
    masks_info, _, _ = state._ENRICH_CACHE[enrich_cache_key]

    df_records = []
    for m in masks_info:
        tax = m.get("taxonomy", {})
        bl = m.get("bleaching", {})
        df_records.append({
            "ID": f"#{m['id']}",
            "Taxon Genus": tax.get("genus", "Coral"),
            "Growth Form": tax.get("growth_form", "-"),
            "Taxon Conf (%)": round(float(tax.get("confidence_pct", 0)), 1),
            "Condition": "Bleached" if bl.get("is_bleached") else "Healthy",
            "Condition Conf (%)": round(float(bl.get("confidence_pct", 0)), 1),
            "Area (%)": f"{m.get('area_pct', 0.0)}%",
            "Area (px)": m.get("area_px", m.get("area", 0)),
            "IoU Confidence": round(float(m["predicted_iou"]), 3),
        })

    df = pd.DataFrame(df_records)
    csv_str = df.to_csv(index=False)
    return Response(content=csv_str, media_type="text/csv")
