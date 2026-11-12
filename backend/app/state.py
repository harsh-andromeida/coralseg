"""
Shared state, in-memory caches, synchronization primitives, and processor singletons
for ReefVision FastAPI application.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image

# Ensure backend root is on sys.path so 'core' can be imported cleanly
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.models import (
    load_coralscop_model,
    load_bioclip_model,
    load_bleaching_model,
    unload_all_models_from_ram,
    are_models_loaded_in_ram,
)
from core.batch import BatchProcessor
from core.video import VideoProcessor

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
    """Retrieves or creates a thread lock for a given segmentation cache key."""
    with _SEG_LOCKS_MUTEX:
        if cache_key not in _SEG_LOCKS:
            _SEG_LOCKS[cache_key] = threading.Lock()
        return _SEG_LOCKS[cache_key]


# Download progress tracker
_DOWNLOAD_PROGRESS: Dict[str, Any] = {
    "is_downloading": False,
    "overall_pct": 0,
    "current_file": "",
    "detail": "",
}

# Video storage directories
VIDEO_STORAGE_DIR = os.path.join(BACKEND_DIR, "uploads", "videos")
VIDEO_EXPORTS_DIR = os.path.join(BACKEND_DIR, "exports", "video")
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_EXPORTS_DIR, exist_ok=True)


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


# Processor singletons
batch_processor: Optional[BatchProcessor] = None
video_processor: Optional[VideoProcessor] = None
_BATCH_PROCESSOR: Optional[BatchProcessor] = None
_VIDEO_PROCESSOR: Optional[VideoProcessor] = None


def init_processors() -> tuple[BatchProcessor, VideoProcessor]:
    """Initializes global BatchProcessor and VideoProcessor singletons."""
    global batch_processor, video_processor, _BATCH_PROCESSOR, _VIDEO_PROCESSOR

    batch_processor = BatchProcessor(
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
    _BATCH_PROCESSOR = batch_processor

    video_processor = VideoProcessor(
        sam_loader=lambda: load_coralscop_model(device=_ACTIVE_DEVICE_PREF),
        bioclip_loader=lambda: load_bioclip_model(device=_ACTIVE_DEVICE_PREF),
        bleaching_loader=lambda: load_bleaching_model(device=_ACTIVE_DEVICE_PREF),
        cache_dir=VIDEO_EXPORTS_DIR,
    )
    _VIDEO_PROCESSOR = video_processor

    return batch_processor, video_processor


# Initialize processors on module import
init_processors()
