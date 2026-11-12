"""
ReefVision FastAPI Engine - Autonomous multi-model coral reef instance segmentation,
taxonomy classification, and bleaching assessment.
"""

import os
import sys
import io
import glob
import json
import base64
import time
import datetime
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Ensure backend root is on sys.path so 'core' can be imported cleanly
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.device import get_device_info, get_optimal_device
from core.models import (
    check_models_download_status,
    download_all_models,
    delete_model_weights,
    load_coralscop_model,
    load_bioclip_model,
    load_bleaching_model,
    unload_all_models_from_ram,
    are_models_loaded_in_ram,
    FOUNDATION_MODEL_SPECS,
    get_cached_model_path,
    HF_SAM_ENCODER_FILE,
    HF_SAM_DECODER_FILE,
)
from core.segmentation import run_segmentation
from core.taxonomy import enrich_masks_with_taxonomy_and_bleaching
from core.visualization import create_segmentation_overlay, generate_distinct_colors
from core.export import build_coco_json, create_coco_dataset_zip, export_dataset_to_directory
from core.batch import BatchProcessor
from core.interactive_segmentation import (
    SAM_SESSION,
    mask_to_polygon,
    polygon_to_mask,
    calculate_polygon_metrics,
)

from core.video import (
    VideoProcessor,
    VideoJobState,
    render_video_overlay,
    export_annotated_video,
    export_transect_csv,
    export_coco_video,
)
from core.video.preprocessor import standardize_video_to_25fps

from app.schemas import (
    DevicePreferenceRequest,
    ModelStatusResponse,
    ModelDownloadRequest,
    ModelDeleteRequest,
    ModelDeleteResponse,
    SegmentRequest,
    EnrichRequest,
    OverlayRequest,
    ExportCocoRequest,
    SampleItem,
    BatchStartRequest,
    BatchPrioritizeRequest,
    BatchStatusResponse,
    ExportFolderRequest,
    ExportFolderResponse,
    PromptPoint,
    SegmentPromptRequest,
    SegmentPromptWarmupRequest,
    EditedSegmentItem,
    SaveAnnotationsRequest,
    VideoProcessRequest,
    VideoStatusResponse,
    VideoTrackItem,
)

app = FastAPI(
    title="Andromeida Reef Vision Studio API",
    version="2.0.0",
    description="FastAPI Backend for Coral Reef Segmentation, BioCLIP Taxonomy, and Bleaching Detection",
)

# Enable CORS for Tauri desktop and Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for images and segmentation session cache
_IMAGE_STORE: Dict[str, Image.Image] = {}
_BASE_SEG_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_ENRICH_CACHE: Dict[str, Any] = {}
_OVERLAY_CACHE: Dict[str, str] = {}
_ACTIVE_DEVICE_PREF: str = "auto"
_DEFAULT_CONCURRENCY: int = 1
_SEG_LOCKS: Dict[str, threading.Lock] = {}
_SEG_LOCKS_MUTEX = threading.Lock()


def _get_seg_lock(cache_key: str) -> threading.Lock:
    with _SEG_LOCKS_MUTEX:
        if cache_key not in _SEG_LOCKS:
            _SEG_LOCKS[cache_key] = threading.Lock()
        return _SEG_LOCKS[cache_key]


# Initialize global background BatchProcessor
_BATCH_PROCESSOR = BatchProcessor(
    image_store=_IMAGE_STORE,
    base_seg_cache=_BASE_SEG_CACHE,
    enrich_cache=_ENRICH_CACHE,
    overlay_cache=_OVERLAY_CACHE,
    model_loader_sam=load_coralscop_model,
    model_loader_bioclip=load_bioclip_model,
    model_loader_bleaching=load_bleaching_model,
    get_active_device=lambda: _ACTIVE_DEVICE_PREF,
    model_unloader=unload_all_models_from_ram,
    models_loaded_checker=are_models_loaded_in_ram,
)

# Video storage directories and VideoProcessor
VIDEO_STORAGE_DIR = os.path.join(BACKEND_DIR, "uploads", "videos")
VIDEO_EXPORTS_DIR = os.path.join(BACKEND_DIR, "exports", "video")
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_EXPORTS_DIR, exist_ok=True)

_VIDEO_PROCESSOR = VideoProcessor(
    sam_loader=lambda: load_coralscop_model(device=_ACTIVE_DEVICE_PREF),
    bioclip_loader=lambda: load_bioclip_model(device=_ACTIVE_DEVICE_PREF),
    bleaching_loader=lambda: load_bleaching_model(device=_ACTIVE_DEVICE_PREF),
    cache_dir=VIDEO_EXPORTS_DIR,
)

_DOWNLOAD_PROGRESS: Dict[str, Any] = {
    "is_downloading": False,
    "overall_pct": 0,
    "current_file": "",
    "detail": "",
}


def _resolve_demo_dirs() -> List[str]:
    """Finds demo_images directories relative to repo or backend."""
    candidates = [
        str(BACKEND_DIR.parent / "demo_images"),
        str(BACKEND_DIR / "demo_images"),
        str(Path.cwd() / "demo_images"),
    ]
    res = []
    for c in candidates:
        if os.path.isdir(c):
            res.append(c)
            sub = os.path.join(c, "unlabeled_samples")
            if os.path.isdir(sub):
                res.append(sub)
    return res


@app.get("/api/health")
def get_health():
    """Liveness probe and active hardware acceleration provider."""
    dev_info = get_device_info(_ACTIVE_DEVICE_PREF)
    return {
        "status": "online",
        "app": "Andromeida Reef Vision Studio",
        "device": dev_info,
    }


@app.get("/api/device/info")
def get_device():
    """Returns active device information."""
    return get_device_info(_ACTIVE_DEVICE_PREF)


@app.post("/api/device/select")
def select_device(req: DevicePreferenceRequest):
    """Updates device preference (auto, cuda, cpu) and batch concurrency."""
    global _ACTIVE_DEVICE_PREF, _DEFAULT_CONCURRENCY
    old_pref = _ACTIVE_DEVICE_PREF
    _ACTIVE_DEVICE_PREF = req.preference.lower().strip()
    if old_pref != _ACTIVE_DEVICE_PREF:
        unload_all_models_from_ram()
    if req.concurrency:
        _DEFAULT_CONCURRENCY = max(1, min(int(req.concurrency), 4))
    info = get_device_info(_ACTIVE_DEVICE_PREF)
    info["concurrency"] = _DEFAULT_CONCURRENCY
    return info


@app.get("/api/models/status")
def get_models_status():
    """Checks download and cache status of foundation models."""
    all_ready, status_list = check_models_download_status()
    dev_info = get_device_info(_ACTIVE_DEVICE_PREF)
    return {
        "all_downloaded": all_ready,
        "models": status_list,
        "active_device": dev_info.get("device", "CPUExecutionProvider"),
        "device_info": dev_info,
    }


@app.get("/api/models/download/progress")
def get_download_progress():
    """Pollable download progress."""
    return _DOWNLOAD_PROGRESS


@app.post("/api/models/download")
def trigger_model_download(
    background_tasks: BackgroundTasks,
    req: Optional[ModelDownloadRequest] = None,
):
    """Triggers downloading or force re-downloading foundation models in background."""
    global _DOWNLOAD_PROGRESS
    if _DOWNLOAD_PROGRESS["is_downloading"]:
        return {"status": "in_progress", "progress": _DOWNLOAD_PROGRESS}

    force = req.force if req else False
    target_files = None
    if req:
        if req.filenames:
            target_files = list(req.filenames)
        elif req.filename:
            target_files = [req.filename]

    _DOWNLOAD_PROGRESS = {
        "is_downloading": True,
        "overall_pct": 0,
        "current_file": "Connecting to Hugging Face...",
        "detail": "Initializing force re-download..." if force else "Initializing download manager",
        "error": None,
    }

    def _download_task():
        global _DOWNLOAD_PROGRESS
        try:
            def _cb(info: dict):
                pct = int(info["overall_fraction"] * 100)
                curr_mb = info["file_bytes"] / (1024 * 1024)
                tot_mb = (info["file_total"] or info["file_bytes"]) / (1024 * 1024)
                _DOWNLOAD_PROGRESS.update({
                    "is_downloading": True,
                    "overall_pct": pct,
                    "current_file": f"[{info['file_index']}/{info['total_files']}] {info['name']}",
                    "detail": f"{curr_mb:.1f} MB / {tot_mb:.1f} MB ({pct}%)",
                })

            download_all_models(
                progress_callback=_cb,
                force_download=force,
                target_files=target_files,
            )

            # Warm up sessions for available models
            try:
                load_coralscop_model(device=_ACTIVE_DEVICE_PREF)
            except Exception:
                pass
            try:
                load_bioclip_model(device=_ACTIVE_DEVICE_PREF)
            except Exception:
                pass
            try:
                load_bleaching_model(device=_ACTIVE_DEVICE_PREF)
            except Exception:
                pass

            _DOWNLOAD_PROGRESS.update({
                "is_downloading": False,
                "overall_pct": 100,
                "current_file": "Download Complete",
                "detail": "Requested model weights downloaded and verified",
            })
        except Exception as e:
            _DOWNLOAD_PROGRESS.update({
                "is_downloading": False,
                "error": str(e),
                "detail": f"Download failed: {e}",
            })

    background_tasks.add_task(_download_task)
    return {"status": "started", "force": force}


@app.post("/api/models/delete", response_model=ModelDeleteResponse)
def delete_model_endpoint(req: Optional[ModelDeleteRequest] = None):
    """Deletes one or all downloaded foundation models and frees disk space."""
    target_files = None
    target_label = "all models"
    if req and not req.all:
        if req.filenames:
            target_files = list(req.filenames)
            target_label = ", ".join(req.filenames)
        elif req.filename:
            target_files = [req.filename]
            target_label = req.filename

    result = delete_model_weights(target_files=target_files)

    # Invalidate in-memory inference caches and cancel any active batch
    _BATCH_PROCESSOR.cancel_batch()
    with _SEG_LOCKS_MUTEX:
        _BASE_SEG_CACHE.clear()
        _ENRICH_CACHE.clear()
        _OVERLAY_CACHE.clear()

    return {
        "status": "success",
        "message": f"Successfully deleted {target_label}",
        "deleted_files": result["deleted_files"],
        "freed_bytes": result["freed_bytes"],
        "all_downloaded": result["all_downloaded"],
        "models": result["models"],
    }


@app.delete("/api/models", response_model=ModelDeleteResponse)
def delete_models_query(filename: Optional[str] = None):
    """Query-param version of model deletion."""
    req = ModelDeleteRequest(filename=filename, all=(filename is None))
    return delete_model_endpoint(req)


@app.get("/api/samples")
def list_samples() -> List[Dict[str, Any]]:
    """Discovers available coral demo samples."""
    dirs = _resolve_demo_dirs()
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


@app.get("/api/samples/image/{filename}")
def get_sample_image(filename: str):
    """Streams a sample image file."""
    dirs = _resolve_demo_dirs()
    for sdir in dirs:
        target = os.path.join(sdir, filename)
        if os.path.isfile(target):
            with open(target, "rb") as f:
                content = f.read()
            ext = filename.rsplit(".", 1)[-1].lower()
            media_type = "image/png" if ext == "png" else "image/jpeg"
            return Response(content=content, media_type=media_type)
    raise HTTPException(status_code=404, detail=f"Sample '{filename}' not found")


@app.post("/api/images/register")
def register_image_base64(payload: Dict[str, str]):
    """Registers an image in memory via base64."""
    image_name = payload.get("filename", f"upload_{int(time.time())}.png")
    data_str = payload.get("data", "")
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    raw_bytes = base64.b64decode(data_str)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    _IMAGE_STORE[image_name] = pil_img
    return {
        "status": "registered",
        "filename": image_name,
        "width": pil_img.width,
        "height": pil_img.height,
    }


@app.post("/api/images/load-sample")
def load_sample_to_store(payload: Dict[str, str]):
    """Loads a named sample image from disk into memory."""
    filename = payload.get("filename", "")
    dirs = _resolve_demo_dirs()
    for sdir in dirs:
        target = os.path.join(sdir, filename)
        if os.path.isfile(target):
            pil_img = Image.open(target).convert("RGB")
            _IMAGE_STORE[filename] = pil_img
            return {
                "status": "loaded",
                "filename": filename,
                "width": pil_img.width,
                "height": pil_img.height,
            }
    raise HTTPException(status_code=404, detail=f"Sample '{filename}' not found")


@app.post("/api/analysis/segment")
def segment_image(req: SegmentRequest):
    """Runs CoralSCOP SAM segmentation on a registered image."""
    all_ready, _ = check_models_download_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are not downloaded or have been deleted. You must download all models before running segmentation.",
        )

    img_name = req.image_name
    if img_name not in _IMAGE_STORE:
        # If image_base64 is provided in the request, register it automatically
        if req.image_base64:
            data_str = req.image_base64
            if "," in data_str:
                data_str = data_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(data_str)
            _IMAGE_STORE[img_name] = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        else:
            # Check if it exists in demo_images
            dirs = _resolve_demo_dirs()
            found = False
            for sdir in dirs:
                target = os.path.join(sdir, img_name)
                if os.path.isfile(target):
                    _IMAGE_STORE[img_name] = Image.open(target).convert("RGB")
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded in session")

    pil_img = _IMAGE_STORE[img_name]
    cache_key = f"{img_name}_{req.points_per_side}_{req.iou_thresh}_{req.stability_thresh}"

    if cache_key not in _BASE_SEG_CACHE:
        seg_lock = _get_seg_lock(cache_key)
        with seg_lock:
            if cache_key not in _BASE_SEG_CACHE:
                sam_bundle = load_coralscop_model(device=_ACTIVE_DEVICE_PREF)
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
                _BASE_SEG_CACHE[cache_key] = masks_info

    all_candidate_masks = _BASE_SEG_CACHE[cache_key]
    return {
        "status": "success",
        "cache_key": cache_key,
        "total_candidate_masks": len(all_candidate_masks),
        "image_width": pil_img.width,
        "image_height": pil_img.height,
    }


@app.post("/api/analysis/enrich")
def enrich_image(req: EnrichRequest):
    """Enriches filtered masks with BioCLIP taxonomy and NOAA bleaching detection."""
    all_ready, _ = check_models_download_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are not downloaded or have been deleted. You must download all models before running taxonomy enrichment.",
        )

    img_name = req.image_name
    if img_name not in _IMAGE_STORE:
        raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded")

    pil_img = _IMAGE_STORE[img_name]
    img_h, img_w = pil_img.height, pil_img.width
    total_pixels = img_h * img_w

    # Find the most recent base seg cache key for this image
    matching_keys = [k for k in _BASE_SEG_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /segment first")
    base_cache_key = matching_keys[-1]
    all_candidate_masks = _BASE_SEG_CACHE[base_cache_key]

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
    if enrich_cache_key not in _ENRICH_CACHE:
        bioclip = load_bioclip_model(device=_ACTIVE_DEVICE_PREF)
        bleaching = load_bleaching_model(device=_ACTIVE_DEVICE_PREF)

        enriched_masks, full_image_eval, health_summary = enrich_masks_with_taxonomy_and_bleaching(
            image=pil_img,
            masks_info=filtered_masks,
            bioclip_bundle=bioclip,
            bleaching_model=bleaching,
        )
        _ENRICH_CACHE[enrich_cache_key] = (enriched_masks, full_image_eval, health_summary)

    masks_info, full_image_eval, health_summary = _ENRICH_CACHE[enrich_cache_key]

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


@app.post("/api/analysis/overlay")
def render_overlay(req: OverlayRequest):
    """Renders high-contrast segmentation overlay on image canvas."""
    img_name = req.image_name
    if img_name not in _IMAGE_STORE:
        raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded")

    pil_img = _IMAGE_STORE[img_name]

    # Find enriched masks
    matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{img_name}_") and k.endswith(f"_{req.min_area_px}")]
    if not matching_keys:
        # Fallback to any enriched cache for this image
        matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /enrich first")

    enrich_cache_key = matching_keys[-1]
    masks_info, _, _ = _ENRICH_CACHE[enrich_cache_key]

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


@app.get("/api/analysis/polygons/{image_name}")
def get_image_polygons(image_name: str):
    """
    Returns polygon boundaries and metadata for all detected or human-annotated
    coral instances for an image.
    """
    if image_name not in _IMAGE_STORE:
        # Try finding in demo_images
        dirs = _resolve_demo_dirs()
        for sdir in dirs:
            target = os.path.join(sdir, image_name)
            if os.path.isfile(target):
                _IMAGE_STORE[image_name] = Image.open(target).convert("RGB")
                break
        if image_name not in _IMAGE_STORE:
            raise HTTPException(status_code=404, detail=f"Image '{image_name}' not loaded in session")

    pil_img = _IMAGE_STORE[image_name]
    img_h, img_w = pil_img.height, pil_img.width

    matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{image_name}_")]
    masks_info = []
    if matching_keys:
        masks_info, _, _ = _ENRICH_CACHE[matching_keys[-1]]
    else:
        # Check base seg cache
        base_keys = [k for k in _BASE_SEG_CACHE.keys() if k.startswith(f"{image_name}_")]
        if base_keys:
            masks_info = _BASE_SEG_CACHE[base_keys[-1]]

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
    """Helper to retrieve or decode an image into _IMAGE_STORE."""
    if img_name in _IMAGE_STORE:
        return _IMAGE_STORE[img_name]

    if image_data:
        data_str = image_data
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(data_str)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            _IMAGE_STORE[img_name] = pil_img
            return pil_img
        except Exception:
            pass

    # Check demo images and unlabeled samples
    dirs = _resolve_demo_dirs()
    for sdir in dirs:
        target = os.path.join(sdir, img_name)
        if os.path.isfile(target):
            try:
                pil_img = Image.open(target).convert("RGB")
                _IMAGE_STORE[img_name] = pil_img
                return pil_img
            except Exception:
                pass
    return None


@app.post("/api/analysis/segment-prompt")
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

    sam_bundle = load_coralscop_model(device=_ACTIVE_DEVICE_PREF)

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


@app.post("/api/analysis/segment-prompt/warmup")
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

    sam_bundle = load_coralscop_model(device=_ACTIVE_DEVICE_PREF)
    SAM_SESSION.get_or_compute_embeddings(sam_bundle, img_name, pil_img)
    return {"status": "warmed_up", "image_name": img_name}


@app.post("/api/analysis/save-annotations")
def save_manual_annotations(req: SaveAnnotationsRequest):
    """
    Saves user-edited and newly created coral instance annotations for an image.
    Marks all modified annotations as 100% confidence human ground truth.
    Recalculates exact area, percent cover, health metrics, and re-renders overlay.
    """
    img_name = req.image_name
    if img_name not in _IMAGE_STORE:
        if req.image_data and req.image_data.startswith("data:"):
            header, b64_str = req.image_data.split(",", 1)
            raw_bytes = base64.b64decode(b64_str)
            _IMAGE_STORE[img_name] = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        else:
            # Check demo images
            dirs = _resolve_demo_dirs()
            for sdir in dirs:
                target = os.path.join(sdir, img_name)
                if os.path.isfile(target):
                    _IMAGE_STORE[img_name] = Image.open(target).convert("RGB")
                    break
        if img_name not in _IMAGE_STORE:
            raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded in session")

    pil_img = _IMAGE_STORE[img_name]
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
    cache_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not cache_keys:
        cache_keys = [f"{img_name}_16_0.5_0.5_100"]
    for k in cache_keys:
        _ENRICH_CACHE[k] = (masks_info, {}, health_summary)

    base_cache_keys = [k for k in _BASE_SEG_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not base_cache_keys:
        base_cache_keys = [f"{img_name}_16_0.5_0.5"]
    for bk in base_cache_keys:
        _BASE_SEG_CACHE[bk] = masks_info

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
    _OVERLAY_CACHE[img_name] = overlay_data_url

    # Update batch processor results if present
    if hasattr(_BATCH_PROCESSOR, "_results") and img_name in _BATCH_PROCESSOR._results:
        _BATCH_PROCESSOR._results[img_name].update({
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


@app.post("/api/analysis/export/coco")
def export_coco(req: ExportCocoRequest):
    """Builds and returns standard COCO JSON format."""
    img_name = req.image_name
    if img_name not in _IMAGE_STORE:
        raise HTTPException(status_code=404, detail=f"Image '{img_name}' not loaded")

    pil_img = _IMAGE_STORE[img_name]
    matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{img_name}_") and k.endswith(f"_{req.min_area_px}")]
    if not matching_keys:
        matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /enrich first")

    enrich_cache_key = matching_keys[-1]
    masks_info, full_image_eval, health_summary = _ENRICH_CACHE[enrich_cache_key]

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


@app.post("/api/analysis/export/csv")
def export_csv(req: ExportCocoRequest):
    """Exports segments table as CSV text."""
    img_name = req.image_name
    matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{img_name}_") and k.endswith(f"_{req.min_area_px}")]
    if not matching_keys:
        matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{img_name}_")]
    if not matching_keys:
        raise HTTPException(status_code=400, detail="Must run /enrich first")

    enrich_cache_key = matching_keys[-1]
    masks_info, _, _ = _ENRICH_CACHE[enrich_cache_key]

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


# ---------------------------------------------------------
# Background Bulk Processing & Structured Dataset Endpoints
# ---------------------------------------------------------

@app.post("/api/batch/start")
def start_batch_processing(req: BatchStartRequest):
    """Starts background bulk processing of multiple images."""
    all_ready, _ = check_models_download_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are missing or deleted. Please download all models before starting batch processing.",
        )

    # Ensure all requested images are registered in memory
    dirs = _resolve_demo_dirs()
    for img_name in req.images:
        if img_name not in _IMAGE_STORE:
            found = False
            for sdir in dirs:
                target = os.path.join(sdir, img_name)
                if os.path.isfile(target):
                    _IMAGE_STORE[img_name] = Image.open(target).convert("RGB")
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
    concurrency = req.concurrency if req.concurrency else _DEFAULT_CONCURRENCY

    return _BATCH_PROCESSOR.start_batch(
        images=req.images,
        params=params,
        concurrency=concurrency,
    )


@app.get("/api/batch/status")
def get_batch_status():
    """Polls real-time bulk processing progress across all images."""
    return _BATCH_PROCESSOR.get_status()


@app.post("/api/batch/pause")
def pause_batch_processing():
    """Pauses background batch processing."""
    return _BATCH_PROCESSOR.pause_batch()


@app.post("/api/batch/resume")
def resume_batch_processing():
    """Resumes background batch processing."""
    return _BATCH_PROCESSOR.resume_batch()


@app.post("/api/batch/cancel")
def cancel_batch_processing():
    """Cancels remaining pending images in the active batch and unloads models from RAM."""
    return _BATCH_PROCESSOR.cancel_batch()


@app.post("/api/models/unload")
def unload_models_from_ram_endpoint():
    """Cancels active batches and unloads all models from RAM to free memory."""
    _BATCH_PROCESSOR.cancel_batch()
    return unload_all_models_from_ram()


@app.post("/api/batch/prioritize")
def prioritize_batch_image(req: BatchPrioritizeRequest):
    """Bumps a specific image to the front of the background processing queue."""
    prioritized = _BATCH_PROCESSOR.prioritize_image(req.image_name)
    return {"prioritized": prioritized, "image_name": req.image_name}


@app.get("/api/analysis/result/{image_name}")
def get_analysis_result(image_name: str):
    """
    Returns precomputed results (segments, stats, health, and overlay)
    for a completed image, or its current background queue state.
    """
    # 1. Check BatchProcessor items
    item = _BATCH_PROCESSOR.get_image_result(image_name)
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
    matching_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{image_name}_")]
    if matching_keys:
        enrich_cache_key = matching_keys[-1]
        masks_info, full_image_eval, health_summary = _ENRICH_CACHE[enrich_cache_key]
        pil_img = _IMAGE_STORE.get(image_name)
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

        overlay_b64 = _OVERLAY_CACHE.get(f"{image_name}_default")

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


def _gather_processed_dataset_items(image_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Collects structured telemetry, taxonomy annotations, masks, and metadata
    for all images processed via batch processor, active analysis studio sessions, or caches.
    """
    # 1. Image names from batch processor
    processed_names = _BATCH_PROCESSOR.get_processed_images_list()

    # 2. Image names from _ENRICH_CACHE
    enrich_names = [
        name for name in _IMAGE_STORE.keys()
        if any(k.startswith(f"{name}_") for k in _ENRICH_CACHE.keys())
    ]

    # 3. Image names from _BASE_SEG_CACHE
    seg_names = [
        name for name in _IMAGE_STORE.keys()
        if any(k.startswith(f"{name}_") for k in _BASE_SEG_CACHE.keys())
    ]

    all_target_names = list(dict.fromkeys(processed_names + enrich_names + seg_names))

    # If explicit image_names filter is provided, restrict to them (or include if in store)
    if image_names:
        allowed = set(image_names)
        all_target_names = [n for n in all_target_names if n in allowed]
        for req_name in image_names:
            if req_name in _IMAGE_STORE and req_name not in all_target_names:
                all_target_names.append(req_name)

    if not all_target_names:
        raise HTTPException(
            status_code=400,
            detail="No analyzed images available to export yet. Please analyze at least one image before exporting.",
        )

    processed_items = []
    for name in all_target_names:
        pil_img = _IMAGE_STORE.get(name)
        w = pil_img.width if pil_img else 1024
        h = pil_img.height if pil_img else 768

        # Find raw enriched masks if present
        matching_enrich_keys = [k for k in _ENRICH_CACHE.keys() if k.startswith(f"{name}_")]
        if matching_enrich_keys:
            masks_info, full_image_eval, health_summary = _ENRICH_CACHE[matching_enrich_keys[-1]]
        else:
            matching_base_keys = [k for k in _BASE_SEG_CACHE.keys() if k.startswith(f"{name}_")]
            if matching_base_keys:
                masks_info = _BASE_SEG_CACHE[matching_base_keys[-1]]
            else:
                masks_info = []
            full_image_eval, health_summary = {}, {}

        batch_item = _BATCH_PROCESSOR.get_image_result(name) or {}
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


@app.get("/api/batch/export/default-path")
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


@app.get("/api/batch/export/coco-zip")
def export_batch_coco_dataset(image_name: Optional[str] = None):
    """
    Assembles and streams a complete structured dataset archive (.zip)
    containing COCO JSON annotations, original images, summary.csv, and dataset_summary.json
    for all images processed up to this point.
    """
    img_filter = [image_name] if image_name else None
    processed_items = _gather_processed_dataset_items(image_names=img_filter)

    zip_bytes = create_coco_dataset_zip(processed_items, _IMAGE_STORE)
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


@app.post("/api/batch/export/to-folder", response_model=ExportFolderResponse)
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
        image_store=_IMAGE_STORE,
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


# --- Video Processing Endpoints ---

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads an underwater video (.mp4, .mov, .avi) and probes its metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    allowed_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    clean_name = os.path.basename(file.filename)
    dest_path = os.path.join(VIDEO_STORAGE_DIR, clean_name)

    with open(dest_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    base, ext = os.path.splitext(clean_name)
    if base.endswith("_25fps"):
        std_name = clean_name
    else:
        std_name = f"{base}_25fps.mp4"
    std_path = os.path.join(VIDEO_STORAGE_DIR, std_name)

    try:
        metadata = standardize_video_to_25fps(dest_path, std_path)
        final_name = std_name
        final_path = std_path
    except Exception as e:
        try:
            metadata = _VIDEO_PROCESSOR.probe_video(dest_path)
            final_name = clean_name
            final_path = dest_path
        except Exception as probe_err:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise HTTPException(status_code=400, detail=f"Failed to probe video: {str(probe_err)}")

    return {
        "status": "success",
        "video_name": final_name,
        "video_path": final_path,
        "metadata": metadata,
    }


@app.get("/api/video/sample")
def get_sample_video():
    """Loads a pre-packaged underwater coral reef transect video standardized to 25 FPS."""
    sample_25fps = os.path.join(BACKEND_DIR.parent, "demo_images", "sample_reef_transect_25fps.mp4")
    sample_src = os.path.join(BACKEND_DIR.parent, "demo_images", "sample_reef_transect.mp4")

    clean_name = "sample_reef_transect_25fps.mp4"
    dest_path = os.path.join(VIDEO_STORAGE_DIR, clean_name)
    import shutil

    if os.path.isfile(sample_25fps):
        if not os.path.isfile(dest_path) or os.path.getmtime(sample_25fps) > os.path.getmtime(dest_path):
            shutil.copyfile(sample_25fps, dest_path)
        metadata = _VIDEO_PROCESSOR.probe_video(dest_path)
    elif os.path.isfile(sample_src):
        if not os.path.isfile(dest_path) or os.path.getmtime(sample_src) > os.path.getmtime(dest_path):
            try:
                metadata = standardize_video_to_25fps(sample_src, dest_path)
            except Exception:
                shutil.copyfile(sample_src, dest_path)
                metadata = _VIDEO_PROCESSOR.probe_video(dest_path)
        else:
            metadata = _VIDEO_PROCESSOR.probe_video(dest_path)
    else:
        raise HTTPException(status_code=404, detail="Sample video not found in demo_images directory.")

    return {
        "status": "success",
        "video_name": clean_name,
        "video_path": dest_path,
        "metadata": metadata,
    }


@app.get("/api/video/stream-file/{video_name}")
def stream_video_file(video_name: str):
    """Streams raw video file for HTML5 video player playback with HTTP range support."""
    from fastapi.responses import FileResponse
    clean_base = os.path.basename(video_name)
    target = os.path.join(VIDEO_STORAGE_DIR, clean_base)
    if not os.path.isfile(target):
        # Gracefully handle redundant _25fps_25fps naming
        if "_25fps_25fps" in clean_base:
            normalized_name = clean_base.replace("_25fps_25fps", "_25fps")
            candidate = os.path.join(VIDEO_STORAGE_DIR, normalized_name)
            if os.path.isfile(candidate):
                target = candidate

    if not os.path.isfile(target):
        demo_target = os.path.join(BACKEND_DIR.parent, "demo_images", clean_base)
        if os.path.isfile(demo_target):
            target = demo_target
        elif "_25fps_25fps" in clean_base:
            demo_norm = os.path.join(BACKEND_DIR.parent, "demo_images", clean_base.replace("_25fps_25fps", "_25fps"))
            if os.path.isfile(demo_norm):
                target = demo_norm

    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"Video file '{video_name}' not found.")

    return FileResponse(
        target,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/api/video/process")
def process_video(req: VideoProcessRequest):
    """Starts asynchronous high-FPS benthic tracking and keyframe analysis on a video."""
    all_ready, _ = check_models_download_status()
    if not all_ready:
        raise HTTPException(
            status_code=428,
            detail="Foundation models are not downloaded. Please download all models before running video processing."
        )

    candidate_path = os.path.join(VIDEO_STORAGE_DIR, req.video_name)
    if not os.path.isfile(candidate_path):
        if os.path.isfile(req.video_name):
            candidate_path = req.video_name
        else:
            raise HTTPException(status_code=404, detail=f"Video file '{req.video_name}' not found.")

    try:
        job_id = _VIDEO_PROCESSOR.start_job(
            video_path=candidate_path,
            params={
                "overlap_threshold": req.overlap_threshold,
                "min_stride": req.min_stride,
                "max_stride": req.max_stride,
                "points_per_side": req.points_per_side,
                "iou_match_threshold": req.iou_match_threshold,
                "pred_iou_thresh": req.pred_iou_thresh,
                "stability_thresh": req.stability_thresh,
                "device_pref": _ACTIVE_DEVICE_PREF,
                "tracker_mode": req.tracker_mode or "sam2",
                "model_profile": req.model_profile,
                "model_size": req.model_size,
                "precision": req.precision,
                "worker_count": req.worker_count,
                "export_mp4": req.export_mp4,
            },
        )
        job = _VIDEO_PROCESSOR.get_job(job_id)
        return {
            "status": "started",
            "job_id": job_id,
            "filename": job["filename"],
            "metadata": job["metadata"],
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start video processing: {str(e)}")


@app.get("/api/video/active-status", response_model=Optional[VideoStatusResponse])
def get_active_video_status():
    """Retrieves real-time processing status of the active or most recent video job."""
    job = _VIDEO_PROCESSOR.get_active_job()
    if not job:
        return None

    return VideoStatusResponse(
        job_id=job["job_id"],
        filename=job["filename"],
        state=job["state"],
        current_frame=job["current_frame"],
        total_frames=job["total_frames"],
        progress_pct=job["progress_pct"],
        fps_realtime=job.get("fps_realtime", 0.0),
        tracker_fps=job.get("tracker_fps", 0.0),
        analysis_fps=job.get("analysis_fps", 0.0),
        elapsed_sec=job.get("elapsed_sec", 0.0),
        active_corals_count=job["active_corals_count"],
        total_unique_corals=job["total_unique_corals"],
        bleaching_prevalence_pct=job["bleaching_prevalence_pct"],
        keyframes_count=job["keyframes_count"],
        metadata=job["metadata"],
        error=job["error"],
        time_series=job["time_series"],
        export_paths=job.get("export_paths"),
        export_status=job.get("export_status", "pending"),
        current_stage=job.get("current_stage", "Processing"),
        current_model=job.get("current_model"),
        current_action=job.get("current_action"),
        stage_detail=job.get("stage_detail"),
        logs=job.get("logs", []),
        tracker_mode=job.get("tracker_mode", "sam2"),
    )


@app.get("/api/video/status/{job_id}", response_model=VideoStatusResponse)
def get_video_status(job_id: str):
    """Retrieves real-time processing status, FPS, and time-series survey telemetry."""
    job = _VIDEO_PROCESSOR.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Video job '{job_id}' not found.")

    return VideoStatusResponse(
        job_id=job["job_id"],
        filename=job["filename"],
        state=job["state"],
        current_frame=job["current_frame"],
        total_frames=job["total_frames"],
        progress_pct=job["progress_pct"],
        fps_realtime=job.get("fps_realtime", 0.0),
        tracker_fps=job.get("tracker_fps", 0.0),
        analysis_fps=job.get("analysis_fps", 0.0),
        elapsed_sec=job.get("elapsed_sec", 0.0),
        active_corals_count=job["active_corals_count"],
        total_unique_corals=job["total_unique_corals"],
        bleaching_prevalence_pct=job["bleaching_prevalence_pct"],
        keyframes_count=job["keyframes_count"],
        metadata=job["metadata"],
        error=job["error"],
        time_series=job["time_series"],
        export_paths=job.get("export_paths"),
        export_status=job.get("export_status", "pending"),
        current_stage=job.get("current_stage", "Processing"),
        current_model=job.get("current_model"),
        current_action=job.get("current_action"),
        stage_detail=job.get("stage_detail"),
        logs=job.get("logs", []),
        tracker_mode=job.get("tracker_mode", "sam2"),
    )


@app.post("/api/video/pause/{job_id}")
def pause_video_job(job_id: str):
    success = _VIDEO_PROCESSOR.pause_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not pause job or job not running.")
    return {"status": "paused", "job_id": job_id}


@app.post("/api/video/resume/{job_id}")
def resume_video_job(job_id: str):
    success = _VIDEO_PROCESSOR.resume_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not resume job or job not paused.")
    return {"status": "resumed", "job_id": job_id}


@app.post("/api/video/cancel/{job_id}")
def cancel_video_job(job_id: str):
    success = _VIDEO_PROCESSOR.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"status": "cancelled", "job_id": job_id}


@app.get("/api/video/frame/{job_id}/{frame_idx}")
def get_video_frame_annotations(job_id: str, frame_idx: int):
    """Returns tracking detections and polygon boundaries for a specific frame."""
    data = _VIDEO_PROCESSOR.get_frame_annotation(job_id, frame_idx)
    if data is None:
        raise HTTPException(status_code=404, detail="Frame data not available.")
    return data


@app.get("/api/video/frames/{job_id}")
def get_video_all_frame_annotations(
    job_id: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
):
    """
    Returns tracking detections and polygon boundaries for a range or all processed frames.
    Allows frontend to pre-cache frame annotations in bulk for 60 FPS stutter-free playback.
    """
    job = _VIDEO_PROCESSOR.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")

    frame_map = job.get("frame_tracks_map", {})
    keyframes = set(job.get("keyframe_indices", []))

    result = {}
    if frame_map:
        max_frame = max(frame_map.keys())
        last_f = min(end_frame, max_frame) if end_frame is not None else max_frame

        for f_idx in range(start_frame, last_f + 1):
            if f_idx in frame_map:
                tracks = frame_map[f_idx]
                result[f_idx] = {
                    "frame_idx": f_idx,
                    "is_keyframe": f_idx in keyframes,
                    "tracks": [t.to_dict() for t in tracks],
                }

    return {
        "job_id": job_id,
        "total_frames_available": len(result),
        "frames": result,
    }


@app.get("/api/video/export/{job_id}")
def export_video_result(job_id: str, format: str = "mp4"):
    """Downloads annotated MP4, Transect CSV, or MOT/COCO JSON with on-demand fallback."""
    from fastapi.responses import FileResponse
    job = _VIDEO_PROCESSOR.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")

    export_paths = job.get("export_paths", {})
    fmt = format.lower().strip()
    cache_dir = _VIDEO_PROCESSOR.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    if fmt == "mp4":
        p = export_paths.get("annotated_mp4")
        if not p or not os.path.isfile(p):
            candidate = os.path.join(cache_dir, f"{job_id}_annotated.mp4")
            if os.path.isfile(candidate):
                p = candidate
                job.setdefault("export_paths", {})["annotated_mp4"] = candidate
            elif job.get("frame_tracks_map") and job.get("video_path") and os.path.isfile(job["video_path"]):
                try:
                    export_annotated_video(
                        input_video_path=job["video_path"],
                        output_mp4_path=candidate,
                        frame_tracks_map=job["frame_tracks_map"],
                        max_frames=job.get("current_frame"),
                    )
                    if os.path.isfile(candidate):
                        p = candidate
                        job.setdefault("export_paths", {})["annotated_mp4"] = candidate
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to generate annotated video: {str(e)}")

        if not p or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="Annotated MP4 export not ready. Please wait for frames to process.")
        return FileResponse(p, media_type="video/mp4", filename=os.path.basename(p))

    elif fmt == "csv":
        p = export_paths.get("transect_csv")
        if not p or not os.path.isfile(p):
            candidate = os.path.join(cache_dir, f"{job_id}_transect.csv")
            if os.path.isfile(candidate):
                p = candidate
                job.setdefault("export_paths", {})["transect_csv"] = candidate
            elif job.get("time_series"):
                try:
                    export_transect_csv(
                        output_csv_path=candidate,
                        time_series_records=job["time_series"],
                    )
                    if os.path.isfile(candidate):
                        p = candidate
                        job.setdefault("export_paths", {})["transect_csv"] = candidate
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

        if not p or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="Transect CSV export not ready.")
        return FileResponse(p, media_type="text/csv", filename=os.path.basename(p))

    elif fmt in ("json", "coco", "mot"):
        p = export_paths.get("coco_json")
        if not p or not os.path.isfile(p):
            candidate = os.path.join(cache_dir, f"{job_id}_coco_video.json")
            if os.path.isfile(candidate):
                p = candidate
                job.setdefault("export_paths", {})["coco_json"] = candidate
            elif job.get("frame_tracks_map"):
                try:
                    meta = job.get("metadata", {})
                    # Flatten unique tracks from frame_tracks_map
                    all_tracks_dict = {}
                    for f_tracks in job["frame_tracks_map"].values():
                        for tr in f_tracks:
                            all_tracks_dict[tr.track_id] = tr

                    export_coco_video(
                        output_json_path=candidate,
                        video_filename=job.get("filename", "video.mp4"),
                        width=meta.get("width", 1920),
                        height=meta.get("height", 1080),
                        total_frames=meta.get("total_frames", job.get("total_frames", 1)),
                        fps=meta.get("fps", 25.0),
                        all_tracks=list(all_tracks_dict.values()),
                        frame_tracks_map=job["frame_tracks_map"],
                    )
                    if os.path.isfile(candidate):
                        p = candidate
                        job.setdefault("export_paths", {})["coco_json"] = candidate
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to generate COCO JSON: {str(e)}")

        if not p or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="COCO JSON export not ready.")
        return FileResponse(p, media_type="application/json", filename=os.path.basename(p))
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: mp4, csv, json")


# If static production build directory exists, mount it to serve the single-bundle frontend directly
from fastapi.staticfiles import StaticFiles
DIST_DIR = os.path.join(BACKEND_DIR.parent, "dist")
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static_frontend")
else:
    @app.get("/")
    def index_fallback():
        return {
            "status": "online",
            "message": "ReefVision Studio API Engine is running. Run 'npm run build' to bundle the web UI.",
            "docs": "/docs",
            "api_health": "/api/health",
        }




