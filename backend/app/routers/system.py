"""
System, hardware configuration, and model management router for ReefVision.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks

from core.device import get_device_info
from core.models import (
    check_models_download_status,
    download_all_models,
    delete_model_weights,
    load_coralscop_model,
    load_bioclip_model,
    load_bleaching_model,
    unload_all_models_from_ram,
)
from app.schemas import (
    DevicePreferenceRequest,
    ModelDownloadRequest,
    ModelDeleteRequest,
    ModelDeleteResponse,
)
from app import state

router = APIRouter(tags=["System & Models"])


@router.get("/api/health")
def get_health():
    """Liveness probe and active hardware acceleration provider."""
    dev_info = get_device_info(state._ACTIVE_DEVICE_PREF)
    return {
        "status": "online",
        "app": "Andromeida Reef Vision Studio",
        "device": dev_info,
    }


@router.get("/api/device/info")
def get_device():
    """Returns active device information."""
    return get_device_info(state._ACTIVE_DEVICE_PREF)


@router.post("/api/device/select")
def select_device(req: DevicePreferenceRequest):
    """Updates device preference (auto, cuda, cpu) and batch concurrency."""
    old_pref = state._ACTIVE_DEVICE_PREF
    state._ACTIVE_DEVICE_PREF = req.preference.lower().strip()
    if old_pref != state._ACTIVE_DEVICE_PREF:
        unload_all_models_from_ram()
    if req.concurrency:
        state._DEFAULT_CONCURRENCY = max(1, min(int(req.concurrency), 4))
    info = get_device_info(state._ACTIVE_DEVICE_PREF)
    info["concurrency"] = state._DEFAULT_CONCURRENCY
    return info


@router.get("/api/models/status")
def get_models_status():
    """Checks download and cache status of foundation models."""
    all_ready, status_list = check_models_download_status()
    dev_info = get_device_info(state._ACTIVE_DEVICE_PREF)
    return {
        "all_downloaded": all_ready,
        "models": status_list,
        "active_device": dev_info.get("device", "CPUExecutionProvider"),
        "device_info": dev_info,
    }


@router.get("/api/models/download/progress")
def get_download_progress():
    """Pollable download progress."""
    return state._DOWNLOAD_PROGRESS


@router.post("/api/models/download")
def trigger_model_download(
    background_tasks: BackgroundTasks,
    req: Optional[ModelDownloadRequest] = None,
):
    """Triggers downloading or force re-downloading foundation models in background."""
    if state._DOWNLOAD_PROGRESS["is_downloading"]:
        return {"status": "in_progress", "progress": state._DOWNLOAD_PROGRESS}

    force = req.force if req else False
    target_files = None
    if req:
        if req.filenames:
            target_files = list(req.filenames)
        elif req.filename:
            target_files = [req.filename]

    state._DOWNLOAD_PROGRESS = {
        "is_downloading": True,
        "overall_pct": 0,
        "current_file": "Connecting to Hugging Face...",
        "detail": "Initializing force re-download..." if force else "Initializing download manager",
        "error": None,
    }

    def _download_task():
        try:
            def _cb(info: dict):
                pct = int(info["overall_fraction"] * 100)
                curr_mb = info["file_bytes"] / (1024 * 1024)
                tot_mb = (info["file_total"] or info["file_bytes"]) / (1024 * 1024)
                state._DOWNLOAD_PROGRESS.update({
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
                load_coralscop_model(device=state._ACTIVE_DEVICE_PREF)
            except Exception:
                pass
            try:
                load_bioclip_model(device=state._ACTIVE_DEVICE_PREF)
            except Exception:
                pass
            try:
                load_bleaching_model(device=state._ACTIVE_DEVICE_PREF)
            except Exception:
                pass

            state._DOWNLOAD_PROGRESS.update({
                "is_downloading": False,
                "overall_pct": 100,
                "current_file": "Download Complete",
                "detail": "Requested model weights downloaded and verified",
            })
        except Exception as e:
            state._DOWNLOAD_PROGRESS.update({
                "is_downloading": False,
                "error": str(e),
                "detail": f"Download failed: {e}",
            })

    background_tasks.add_task(_download_task)
    return {"status": "started", "force": force}


@router.post("/api/models/delete", response_model=ModelDeleteResponse)
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
    if state.batch_processor:
        state.batch_processor.cancel_batch()
    with state._SEG_LOCKS_MUTEX:
        state._BASE_SEG_CACHE.clear()
        state._ENRICH_CACHE.clear()
        state._OVERLAY_CACHE.clear()

    return {
        "status": "success",
        "message": f"Successfully deleted {target_label}",
        "deleted_files": result["deleted_files"],
        "freed_bytes": result["freed_bytes"],
        "all_downloaded": result["all_downloaded"],
        "models": result["models"],
    }


@router.delete("/api/models", response_model=ModelDeleteResponse)
def delete_models_query(filename: Optional[str] = None):
    """Query-param version of model deletion."""
    req = ModelDeleteRequest(filename=filename, all=(filename is None))
    return delete_model_endpoint(req)


@router.post("/api/models/unload")
def unload_models_from_ram_endpoint():
    """Cancels active batches and unloads all models from RAM to free memory."""
    if state.batch_processor:
        state.batch_processor.cancel_batch()
    return unload_all_models_from_ram()
