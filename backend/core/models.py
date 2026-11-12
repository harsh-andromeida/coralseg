"""
Model registry, artifact resolution, streaming download manager, and ONNX session loaders.
"""

import os
import sys
import shutil
import threading
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import onnxruntime as ort
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download, try_to_load_from_cache, constants
from huggingface_hub.file_download import repo_folder_name

from core.device import (
    get_optimal_device,
    get_device_mode,
    get_session_options,
    get_cuda_provider_options,
    get_memory_budget,
    can_admit_worker,
)

# Model-specific inference locks for high-throughput concurrency
_SAM_LOCK = threading.Lock()
_BIOCLIP_LOCK = threading.Lock()
_BLEACHING_LOCK = threading.Lock()
_INFERENCE_LOCK = _SAM_LOCK  # Backwards compatibility alias

# Active job reference counter to protect loaded sessions during active jobs
_ACTIVE_JOBS_COUNT = 0
_ACTIVE_JOBS_LOCK = threading.Lock()


def acquire_active_job() -> int:
    """Registers an active job holding model references."""
    global _ACTIVE_JOBS_COUNT
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS_COUNT += 1
        return _ACTIVE_JOBS_COUNT


def release_active_job() -> int:
    """Unregisters an active job holding model references."""
    global _ACTIVE_JOBS_COUNT
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS_COUNT = max(0, _ACTIVE_JOBS_COUNT - 1)
        return _ACTIVE_JOBS_COUNT


def has_active_jobs() -> bool:
    """Checks whether any active jobs currently hold model references."""
    with _ACTIVE_JOBS_LOCK:
        return _ACTIVE_JOBS_COUNT > 0

# Official Hugging Face ONNX repository
HF_ONNX_REPO_ID = "harsh-awasthi/Andromeida-ReefVision-ONNX"
HF_REPO_ID = HF_ONNX_REPO_ID
HF_BLEACHING_FILE = "bleaching_yolo11n.onnx"
HF_BIOCLIP_FILE = "bioclip_visual.onnx"
HF_TAXONOMY_FILE = "coral_taxonomy_embeddings.npy"
HF_TEXT_EMBEDS_FILE = HF_TAXONOMY_FILE
HF_SAM_ENCODER_FILE = "sam_image_encoder.onnx"
HF_SAM_DECODER_FILE = "sam_mask_decoder.onnx"
EXPECTED_MODELS = None  # assigned below after FOUNDATION_MODEL_SPECS

# Global cached sessions
_SAM_BUNDLE: Optional[Dict[str, ort.InferenceSession]] = None
_SAM_CACHE_KEY: Optional[str] = None
_SAM_MTIME: Optional[float] = None
_BIOCLIP_BUNDLE: Optional[Dict[str, Any]] = None
_BIOCLIP_CACHE_KEY: Optional[str] = None
_BLEACHING_SESSION: Optional[ort.InferenceSession] = None
_BLEACHING_CACHE_KEY: Optional[str] = None

FOUNDATION_MODEL_SPECS = [
    {
        "filename": HF_SAM_ENCODER_FILE,
        "name": "Segmentation Model Encoder",
        "task": "Dense Instance Segmentation",
        "approx_size": 359_076_485,
    },
    {
        "filename": HF_SAM_DECODER_FILE,
        "name": "Segmentation Model Decoder",
        "task": "Dense Instance Segmentation",
        "approx_size": 23_808_270,
    },
    {
        "filename": HF_BIOCLIP_FILE,
        "name": "Taxonomical Model",
        "task": "Taxonomy & Growth Form",
        "approx_size": 345_153_901,
    },
    {
        "filename": HF_TAXONOMY_FILE,
        "name": "Taxonomical Model Embeddings",
        "task": "Precomputed Text Embeddings",
        "approx_size": 36_992,
    },
    {
        "filename": HF_BLEACHING_FILE,
        "name": "Bleach Detection Model",
        "task": "Reef Health & Bleaching",
        "approx_size": 6_147_661,
    },
]
EXPECTED_MODELS = FOUNDATION_MODEL_SPECS

MODEL_GROUPS: Dict[str, List[str]] = {
    "sam": [HF_SAM_ENCODER_FILE, HF_SAM_DECODER_FILE],
    "segmentation": [HF_SAM_ENCODER_FILE, HF_SAM_DECODER_FILE],
    "bioclip": [HF_BIOCLIP_FILE, HF_TAXONOMY_FILE],
    "taxonomy": [HF_BIOCLIP_FILE, HF_TAXONOMY_FILE],
    "bleaching": [HF_BLEACHING_FILE],
    "bleach": [HF_BLEACHING_FILE],
}


def get_cached_model_path(filename: str) -> Optional[str]:
    """
    Checks candidate directories and Hugging Face cache for an existing model file.
    Returns the absolute path if found, or None if download is needed.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "onnx_models", filename),
        os.path.join(os.path.dirname(__file__), "..", "models", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", "onnx_models", filename),
        os.path.join(os.getcwd(), "onnx_models", filename),
        os.path.join(os.getcwd(), "models", filename),
    ]
    if getattr(sys, "frozen", False):
        exec_dir = os.path.dirname(sys.executable)
        candidates.insert(0, os.path.join(exec_dir, "models", filename))
        candidates.insert(0, os.path.join(exec_dir, "onnx_models", filename))
        if "Contents/MacOS" in exec_dir:
            bundle_dir = os.path.abspath(os.path.join(exec_dir, "../../.."))
            candidates.insert(0, os.path.join(bundle_dir, "models", filename))

    for path in candidates:
        if os.path.isfile(path):
            return path

    try:
        cached_hf = try_to_load_from_cache(HF_ONNX_REPO_ID, filename)
        if cached_hf is not None and os.path.isfile(cached_hf):
            return cached_hf
    except Exception:
        pass

    return None


def reset_cached_sessions():
    """Clears in-memory ONNX Runtime sessions to allow fresh weight reloads."""
    global _SAM_BUNDLE, _SAM_CACHE_KEY, _SAM_MTIME
    global _BIOCLIP_BUNDLE, _BIOCLIP_CACHE_KEY
    global _BLEACHING_SESSION, _BLEACHING_CACHE_KEY
    _SAM_BUNDLE = None
    _SAM_CACHE_KEY = None
    _SAM_MTIME = None
    _BIOCLIP_BUNDLE = None
    _BIOCLIP_CACHE_KEY = None
    _BLEACHING_SESSION = None
    _BLEACHING_CACHE_KEY = None


def unload_all_models_from_ram() -> Dict[str, Any]:
    """
    Safely closes and unloads all active ONNX Runtime sessions,
    releases text embeddings, and triggers garbage collection to free RAM.
    Fails safely if active jobs hold model references.
    """
    if has_active_jobs():
        return {
            "unloaded": False,
            "message": "Cannot unload models while active video or batch processing jobs are running.",
        }
    reset_cached_sessions()
    import gc
    gc.collect()
    return {"unloaded": True, "message": "All model sessions safely unloaded from RAM"}


def are_models_loaded_in_ram() -> Dict[str, bool]:
    """Returns whether each foundation model is currently loaded in RAM."""
    return {
        "sam_loaded": _SAM_BUNDLE is not None,
        "bioclip_loaded": _BIOCLIP_BUNDLE is not None,
        "bleaching_loaded": _BLEACHING_SESSION is not None,
    }


def find_or_download_onnx_model(
    filename: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, Optional[int], str], None]] = None,
    force_download: bool = False,
) -> str:
    """
    Locates an ONNX model file on local disk or streams it on demand from Hugging Face.
    When force_download is True, bypasses local caches and fetches fresh weights.
    """
    if not force_download:
        cached_path = get_cached_model_path(filename)
        if cached_path is not None:
            if progress_callback is not None:
                size = os.path.getsize(cached_path)
                progress_callback(size, size, filename)
            return cached_path

    class CallbackTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cb = progress_callback
            self._fname = filename
            if self._cb and self.total:
                self._cb(self.n, self.total, self._fname)

        def update(self, n=1):
            res = super().update(n)
            if self._cb and self.total:
                self._cb(self.n, self.total, self._fname)
            return res

    original_tqdm = tqdm.auto.tqdm if hasattr(tqdm, "auto") else tqdm

    try:
        if progress_callback is not None:
            import huggingface_hub.file_download as fd
            fd.tqdm = CallbackTqdm

        local_path = hf_hub_download(
            repo_id=HF_ONNX_REPO_ID,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=False,
            force_download=force_download,
        )
        return local_path
    finally:
        if progress_callback is not None:
            import huggingface_hub.file_download as fd
            fd.tqdm = original_tqdm


def check_models_download_status() -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Evaluates whether all required ONNX weights exist locally.
    """
    status_list = []
    all_ready = True
    for spec in FOUNDATION_MODEL_SPECS:
        path = get_cached_model_path(spec["filename"])
        downloaded = path is not None and os.path.isfile(path)
        actual_size = os.path.getsize(path) if downloaded else spec["approx_size"]
        if not downloaded:
            all_ready = False
        size_str = f"{actual_size / (1024 * 1024):.1f} MB" if actual_size >= 1024 * 1024 else f"{actual_size / 1024:.1f} KB"
        status_list.append({
            "filename": spec["filename"],
            "name": spec["name"],
            "task": spec["task"],
            "approx_size": spec["approx_size"],
            "actual_size": actual_size,
            "size": size_str,
            "downloaded": downloaded,
            "cached": downloaded,
            "path": path,
        })
    return all_ready, status_list


def download_all_models(
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    force_download: bool = False,
    target_files: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Downloads foundation models with progress callbacks.
    When force_download is True, forces fetching from Hugging Face even if cached.
    Supports either an info dictionary callback or a (filename, curr, total) callback.
    """
    if force_download:
        reset_cached_sessions()

    specs = FOUNDATION_MODEL_SPECS
    if target_files:
        expanded_targets = []
        for t in target_files:
            if t in MODEL_GROUPS:
                expanded_targets.extend(MODEL_GROUPS[t])
            else:
                expanded_targets.append(t)
        filtered = [s for s in specs if s["filename"] in expanded_targets]
        if filtered:
            specs = filtered

    downloaded_paths = {}
    total_files = len(specs)
    total_bytes_est = sum(spec["approx_size"] for spec in specs)
    completed_bytes_prior = 0

    for idx, spec in enumerate(specs):
        fname = spec["filename"]
        approx_sz = spec["approx_size"]

        def _cb(curr, total, _f=fname, _idx=idx, _spec=spec, _prior=completed_bytes_prior):
            if progress_callback:
                file_tot = total or approx_sz
                cur_overall = _prior + min(curr, file_tot)
                frac = cur_overall / max(total_bytes_est, 1)
                info = {
                    "file_index": _idx + 1,
                    "total_files": total_files,
                    "name": _spec["name"],
                    "filename": _f,
                    "file_bytes": curr,
                    "file_total": file_tot,
                    "overall_fraction": frac,
                    "status": "completed" if curr >= file_tot else "downloading",
                }
                try:
                    progress_callback(info)
                except TypeError:
                    try:
                        progress_callback(_f, curr, total)
                    except TypeError:
                        progress_callback(curr, total, _f)

        path = find_or_download_onnx_model(
            fname,
            cache_dir=cache_dir,
            progress_callback=_cb,
            force_download=force_download,
        )
        downloaded_paths[fname] = path
        completed_bytes_prior += approx_sz
    return downloaded_paths


def delete_model_weights(target_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Deletes specified local and cached Hugging Face model weights.
    If target_files is None or contains all models, clears all cached weights and HF repo cache.
    Releases all active in-memory ONNX Runtime sessions.
    Fails safely if active jobs hold model references.
    """
    if has_active_jobs():
        all_ready, status_list = check_models_download_status()
        return {
            "deleted_files": [],
            "freed_bytes": 0,
            "all_downloaded": all_ready,
            "models": status_list,
            "error": "Cannot delete models while active video or batch processing jobs are running.",
        }
    reset_cached_sessions()

    specs = FOUNDATION_MODEL_SPECS
    all_filenames = [s["filename"] for s in specs]

    is_delete_all = False
    if target_files is None:
        is_delete_all = True
        targets = list(all_filenames)
    else:
        expanded_targets = []
        for t in target_files:
            if t in MODEL_GROUPS:
                expanded_targets.extend(MODEL_GROUPS[t])
            else:
                expanded_targets.append(t)
        targets = [f for f in expanded_targets if f in all_filenames]
        if set(targets) == set(all_filenames):
            is_delete_all = True

    deleted_files = []
    freed_bytes = 0

    # 1. Local candidate directories check and delete
    for fname in targets:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "onnx_models", fname),
            os.path.join(os.path.dirname(__file__), "..", "models", fname),
            os.path.join(os.path.dirname(__file__), "..", "..", "onnx_models", fname),
            os.path.join(os.getcwd(), "onnx_models", fname),
            os.path.join(os.getcwd(), "models", fname),
        ]
        if getattr(sys, "frozen", False):
            exec_dir = os.path.dirname(sys.executable)
            candidates.insert(0, os.path.join(exec_dir, "models", fname))
            candidates.insert(0, os.path.join(exec_dir, "onnx_models", fname))
            if "Contents/MacOS" in exec_dir:
                bundle_dir = os.path.abspath(os.path.join(exec_dir, "../../.."))
                candidates.insert(0, os.path.join(bundle_dir, "models", fname))

        for cand in candidates:
            if os.path.isfile(cand):
                try:
                    sz = os.path.getsize(cand)
                    os.remove(cand)
                    deleted_files.append(cand)
                    freed_bytes += sz
                except OSError:
                    pass

    # 2. Hugging Face cache deletion
    hf_cache_dir = getattr(constants, "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    repo_folder = repo_folder_name(repo_id=HF_ONNX_REPO_ID, repo_type="model")
    repo_cache_path = os.path.join(hf_cache_dir, repo_folder)

    if is_delete_all:
        if os.path.isdir(repo_cache_path):
            try:
                for root, _, files in os.walk(repo_cache_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        if not os.path.islink(fp):
                            try:
                                freed_bytes += os.path.getsize(fp)
                            except OSError:
                                pass
                shutil.rmtree(repo_cache_path, ignore_errors=True)
                deleted_files.append(repo_cache_path)
            except Exception:
                pass
    else:
        if os.path.isdir(repo_cache_path):
            snapshots_dir = os.path.join(repo_cache_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                for snap_name in os.listdir(snapshots_dir):
                    snap_path = os.path.join(snapshots_dir, snap_name)
                    if os.path.isdir(snap_path):
                        for fname in targets:
                            file_link = os.path.join(snap_path, fname)
                            if os.path.islink(file_link) or os.path.isfile(file_link):
                                if os.path.islink(file_link):
                                    blob_target = os.path.realpath(file_link)
                                    if os.path.isfile(blob_target):
                                        try:
                                            sz = os.path.getsize(blob_target)
                                            os.remove(blob_target)
                                            freed_bytes += sz
                                            deleted_files.append(blob_target)
                                        except OSError:
                                            pass
                                    try:
                                        os.unlink(file_link)
                                        deleted_files.append(file_link)
                                    except OSError:
                                        pass
                                elif os.path.isfile(file_link):
                                    try:
                                        sz = os.path.getsize(file_link)
                                        os.remove(file_link)
                                        freed_bytes += sz
                                        deleted_files.append(file_link)
                                    except OSError:
                                        pass

            try:
                from huggingface_hub import scan_cache_dir
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if repo.repo_id == HF_ONNX_REPO_ID:
                        for rev in repo.revisions:
                            for f in rev.files:
                                if f.file_name in targets:
                                    if os.path.islink(f.file_path) or os.path.isfile(f.file_path):
                                        try:
                                            os.remove(f.file_path)
                                        except OSError:
                                            pass
                                    if os.path.isfile(f.blob_path):
                                        try:
                                            sz = os.path.getsize(f.blob_path)
                                            os.remove(f.blob_path)
                                            freed_bytes += sz
                                            deleted_files.append(str(f.blob_path))
                                        except OSError:
                                            pass
            except Exception:
                pass

    all_ready, status_list = check_models_download_status()
    return {
        "deleted_files": list(set(deleted_files)),
        "freed_bytes": freed_bytes,
        "all_downloaded": all_ready,
        "models": status_list,
    }


def download_coralscop_checkpoint(cache_dir: Optional[str] = None) -> str:
    """Ensures CoralSCOP SAM encoder is available locally."""
    return find_or_download_onnx_model(HF_SAM_ENCODER_FILE, cache_dir=cache_dir)


def load_coralscop_model(
    device: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, ort.InferenceSession]:
    """
    Loads and returns CoralSCOP SAM ViT-B ONNX sessions (Encoder & Decoder).
    """
    global _SAM_BUNDLE, _SAM_CACHE_KEY, _SAM_MTIME
    active_provider = get_optimal_device(device if device is not None else "auto")
    active_mode = get_device_mode(device if device is not None else "auto")
    sess_options = get_session_options(device if device is not None else "auto")
    cache_key = f"{active_provider}:{active_mode}"

    enc_path = find_or_download_onnx_model(HF_SAM_ENCODER_FILE, progress_callback=progress_callback)
    dec_path = find_or_download_onnx_model(HF_SAM_DECODER_FILE, progress_callback=progress_callback)
    current_mtime = os.path.getmtime(dec_path) if os.path.exists(dec_path) else 0.0

    if _SAM_BUNDLE is not None and _SAM_CACHE_KEY == cache_key and _SAM_MTIME == current_mtime:
        return _SAM_BUNDLE

    _SAM_CACHE_KEY = cache_key
    _SAM_MTIME = current_mtime

    # Attempt native GPU acceleration first without silent CPU fallback hiding initialization issues
    if active_provider == "CUDAExecutionProvider":
        try:
            cuda_providers = [("CUDAExecutionProvider", get_cuda_provider_options())]
            encoder_sess = ort.InferenceSession(enc_path, sess_options=sess_options, providers=cuda_providers)
            decoder_sess = ort.InferenceSession(dec_path, sess_options=sess_options, providers=cuda_providers)
            actual_prov = encoder_sess.get_providers()[0]
            print(f"  ⚡ [Model] CoralSCOP SAM sessions successfully loaded on NVIDIA CUDA GPU ({actual_prov})!")
            _SAM_BUNDLE = {
                "encoder": encoder_sess,
                "decoder": decoder_sess,
            }
            return _SAM_BUNDLE
        except Exception as cuda_init_err:
            print(f"  ⚠️ [Model] CoralSCOP SAM CUDA provider error ({cuda_init_err}); falling back to CPUExecutionProvider...")

    providers = [active_provider]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    encoder_sess = ort.InferenceSession(enc_path, sess_options=sess_options, providers=providers)
    decoder_sess = ort.InferenceSession(dec_path, sess_options=sess_options, providers=providers)
    actual_prov = encoder_sess.get_providers()[0]
    print(f"[Model] CoralSCOP SAM sessions initialized with provider: {actual_prov} (requested: {active_provider})")

    _SAM_BUNDLE = {
        "encoder": encoder_sess,
        "decoder": decoder_sess,
    }
    return _SAM_BUNDLE


def load_bioclip_model(
    device: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Loads ReefNet BioCLIP ONNX session and pre-computed text taxonomy embeddings.
    """
    global _BIOCLIP_BUNDLE, _BIOCLIP_CACHE_KEY
    active_provider = get_optimal_device(device if device is not None else "auto")
    active_mode = get_device_mode(device if device is not None else "auto")
    sess_options = get_session_options(device if device is not None else "auto")
    cache_key = f"{active_provider}:{active_mode}"

    if _BIOCLIP_BUNDLE is not None and _BIOCLIP_CACHE_KEY == cache_key:
        return _BIOCLIP_BUNDLE

    _BIOCLIP_CACHE_KEY = cache_key
    model_path = find_or_download_onnx_model(HF_BIOCLIP_FILE, progress_callback=progress_callback)
    embeds_path = find_or_download_onnx_model(HF_TAXONOMY_FILE, progress_callback=progress_callback)
    text_features = np.load(embeds_path).astype(np.float32)

    if active_provider == "CUDAExecutionProvider":
        try:
            cuda_providers = [("CUDAExecutionProvider", get_cuda_provider_options())]
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=cuda_providers)
            actual_prov = session.get_providers()[0]
            print(f"  ⚡ [Model] BioCLIP session successfully loaded on NVIDIA CUDA GPU ({actual_prov})!")
            _BIOCLIP_BUNDLE = {
                "session": session,
                "text_features": text_features,
            }
            return _BIOCLIP_BUNDLE
        except Exception as cuda_init_err:
            print(f"  ⚠️ [Model] BioCLIP CUDA provider error ({cuda_init_err}); falling back to CPUExecutionProvider...")

    providers = [active_provider]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    actual_prov = session.get_providers()[0]
    print(f"[Model] BioCLIP session initialized with provider: {actual_prov} (requested: {active_provider})")

    _BIOCLIP_BUNDLE = {
        "session": session,
        "text_features": text_features,
    }
    return _BIOCLIP_BUNDLE


def load_bleaching_model(
    device: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> ort.InferenceSession:
    """
    Loads NMFS-OSI NOAA coral bleaching classifier ONNX session.
    Uses CUDA on NVIDIA, DirectML on Windows, or CPU fallback with configured threading.
    """
    global _BLEACHING_SESSION, _BLEACHING_CACHE_KEY
    active_provider = get_optimal_device(device if device is not None else "auto")
    active_mode = get_device_mode(device if device is not None else "auto")
    sess_options = get_session_options(device if device is not None else "auto")
    cache_key = f"{active_provider}:{active_mode}"

    if _BLEACHING_SESSION is not None and _BLEACHING_CACHE_KEY == cache_key:
        return _BLEACHING_SESSION

    _BLEACHING_CACHE_KEY = cache_key
    pt_path = find_or_download_onnx_model(HF_BLEACHING_FILE, progress_callback=progress_callback)

    if active_provider == "CUDAExecutionProvider":
        try:
            cuda_providers = [("CUDAExecutionProvider", get_cuda_provider_options())]
            _BLEACHING_SESSION = ort.InferenceSession(pt_path, sess_options=sess_options, providers=cuda_providers)
            actual_prov = _BLEACHING_SESSION.get_providers()[0]
            print(f"  ⚡ [Model] Bleaching YOLO11n session successfully loaded on NVIDIA CUDA GPU ({actual_prov})!")
            return _BLEACHING_SESSION
        except Exception as cuda_init_err:
            print(f"  ⚠️ [Model] Bleaching YOLO CUDA provider error ({cuda_init_err}); falling back to CPUExecutionProvider...")

    providers = [active_provider]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    _BLEACHING_SESSION = ort.InferenceSession(pt_path, sess_options=sess_options, providers=providers)
    actual_prov = _BLEACHING_SESSION.get_providers()[0]
    print(f"[Model] Bleaching YOLO11n session initialized with provider: {actual_prov} (requested: {active_provider})")
    return _BLEACHING_SESSION


def reserve_memory_for_auxiliary_models(device_pref: str = "auto") -> Tuple[bool, str]:
    """
    Ensures sufficient host and device memory is reserved for discovery (CoralSCOP SAM)
    and classification (BioCLIP + YOLO Bleaching) before admitting tracking workers.
    Est. discovery + classification footprint: ~1.8 GiB device / host RAM.
    """
    est_bytes = int(1.8 * 1024**3)
    return can_admit_worker(worker_peak_bytes=est_bytes)


def preload_all_foundation_models(device: Optional[str] = None) -> Dict[str, Any]:
    """
    Warms up and preloads foundation models into resident memory/VRAM:
    1. SAM ViT-B Instance Segmentation (Encoder + Decoder)
    2. ReefNet BioCLIP Taxonomy Classifier & Embeddings
    3. NMFS-OSI NOAA Bleaching Classifier
    """
    sam_bundle = load_coralscop_model(device=device)
    bioclip_bundle = load_bioclip_model(device=device)
    bleaching_session = load_bleaching_model(device=device)
    return {
        "sam": sam_bundle,
        "bioclip": bioclip_bundle,
        "bleaching": bleaching_session,
    }
