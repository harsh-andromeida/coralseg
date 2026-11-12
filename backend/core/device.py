"""
Hardware device introspection and ONNX Runtime execution provider resolution.
Separates PyTorch runtime capabilities from ONNX Runtime providers,
enforces strict accelerator failure vs auto fallback, manages CPU thread budgets,
and exposes accurate host and accelerator memory budgets.
"""

import os
import sys
import site
import platform
import psutil
from typing import Dict, Any, Tuple, Optional, List

# ---------------------------------------------------------------------------
# Dynamic CUDA / cuDNN Path Configuration (Linux, Colab, Windows)
# ---------------------------------------------------------------------------
def _setup_cuda_environment():
    """
    Auto-discovers and configures NVIDIA CUDA and cuDNN shared libraries
    from Python site-packages (e.g. nvidia-* and torch/lib) and system paths.
    Ensures onnxruntime-gpu can locate libcudnn, libcublas, libcufft, etc.
    without manual LD_LIBRARY_PATH exports in Google Colab, Linux, and Windows.
    Must run before importing onnxruntime.
    """
    candidate_dirs: List[str] = []

    # 1. Inspect Python site-packages and dist-packages for nvidia wheels
    site_dirs: List[str] = []
    if hasattr(site, "getsitepackages"):
        try:
            site_dirs.extend(site.getsitepackages())
        except Exception:
            pass
    if hasattr(site, "getusersitepackages"):
        try:
            site_dirs.append(site.getusersitepackages())
        except Exception:
            pass
    site_dirs.extend([p for p in sys.path if "site-packages" in p or "dist-packages" in p])

    for s_dir in set(site_dirs):
        if not os.path.isdir(s_dir):
            continue
        nvidia_root = os.path.join(s_dir, "nvidia")
        if os.path.isdir(nvidia_root):
            for sub in os.listdir(nvidia_root):
                for folder in ("lib", "bin"):
                    p = os.path.join(nvidia_root, sub, folder)
                    if os.path.isdir(p):
                        candidate_dirs.append(p)
        torch_lib = os.path.join(s_dir, "torch", "lib")
        if os.path.isdir(torch_lib):
            candidate_dirs.append(torch_lib)

    # 2. Standard Linux / Colab system CUDA installations
    for p in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/bin",
        "/opt/cuda/lib64",
    ):
        if os.path.isdir(p):
            candidate_dirs.append(p)

    try:
        if os.path.isdir("/usr/local"):
            for entry in os.listdir("/usr/local"):
                if entry.startswith("cuda-"):
                    c_path = os.path.join("/usr/local", entry, "lib64")
                    if os.path.isdir(c_path):
                        candidate_dirs.append(c_path)
    except Exception:
        pass

    # 3. Apply discovered library paths to dynamic loader
    if sys.platform.startswith("linux"):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = [p for p in candidate_dirs if p not in current_ld]
        if new_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + (f":{current_ld}" if current_ld else "")

        CUDA_LIB_PREFIXES = (
            "libcudart.so",
            "libcublas.so",
            "libcublasLt.so",
            "libcudnn.so",
            "libcufft.so",
            "libcurand.so",
            "libcusolver.so",
            "libcusparse.so",
        )
        import ctypes
        for c_dir in candidate_dirs:
            if not os.path.isdir(c_dir):
                continue
            try:
                files = os.listdir(c_dir)
            except Exception:
                continue
            for fname in sorted(files):
                if any(fname.startswith(prefix) for prefix in CUDA_LIB_PREFIXES):
                    try:
                        ctypes.CDLL(os.path.join(c_dir, fname), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass

    elif sys.platform.startswith("win"):
        current_path = os.environ.get("PATH", "")
        new_paths = [p for p in candidate_dirs if p not in current_path]
        if new_paths:
            os.environ["PATH"] = ";".join(new_paths) + (f";{current_path}" if current_path else "")
        if hasattr(os, "add_dll_directory"):
            for c_dir in candidate_dirs:
                try:
                    os.add_dll_directory(c_dir)
                except Exception:
                    pass


# Initialize environment paths before loading onnxruntime
_setup_cuda_environment()

import onnxruntime as ort

GPU_PROVIDERS = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "TensorrtExecutionProvider",
]

_COREML_VERIFIED: Optional[bool] = None
_COREML_LATENCY_MS: Optional[float] = None


def check_torch_capabilities() -> Dict[str, Any]:
    """
    Introspects PyTorch hardware acceleration independently of ONNX Runtime.
    Inspects CUDA, Apple Silicon MPS, compute capability, and native precision support.
    """
    has_torch = False
    has_cuda = False
    has_mps = False
    gpu_name = "CPU"
    compute_capability = None
    native_bf16 = False
    cuda_fp16 = False
    mps_fp32 = False
    mps_fp16 = False

    try:
        import torch
        has_torch = True
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            cuda_fp16 = True
            # BF16 ONLY when supported without emulation
            try:
                native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:
                native_bf16 = torch.cuda.is_bf16_supported()
        elif has_mps:
            gpu_name = f"Apple Silicon ({platform.processor() or 'Metal MPS'})"
            mps_fp32 = True
            # FP16 on MPS is exposed as a tested profile, not the initial correctness default
            mps_fp16 = True
    except Exception:
        pass

    # Precision recommendations:
    # CUDA: BF16 if native without emulation, otherwise FP16
    # MPS: FP32 default for correctness; FP16 as optional tested profile
    # CPU: FP32
    if has_cuda:
        rec_sam_precision = "bfloat16" if native_bf16 else "float16"
    elif has_mps:
        rec_sam_precision = "float32"
    else:
        rec_sam_precision = "float32"

    return {
        "torch_available": has_torch,
        "cuda_available": has_cuda,
        "mps_available": has_mps,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "native_precision": {
            "cuda_bf16_native": native_bf16,
            "cuda_fp16": cuda_fp16,
            "mps_fp32": mps_fp32,
            "mps_fp16_profile": mps_fp16,
            "recommended_sam_precision": rec_sam_precision,
        },
    }


def verify_coreml_operational() -> Tuple[bool, Optional[float]]:
    """
    Attempts to verify CoreMLExecutionProvider by executing a minimal test session.
    Returns (is_operational, latency_ms). Caches result across calls.
    CoreML availability in get_available_providers() alone is insufficient.
    """
    global _COREML_VERIFIED, _COREML_LATENCY_MS
    if _COREML_VERIFIED is not None:
        return _COREML_VERIFIED, _COREML_LATENCY_MS

    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" not in available:
        _COREML_VERIFIED = False
        _COREML_LATENCY_MS = None
        return False, None

    # Test CoreML execution with a tiny synthetic ONNX model or available model
    import time
    import numpy as np

    try:
        # Check if bleaching model or other cached ONNX exists to test session execution
        from core.models import get_cached_model_path, HF_BLEACHING_FILE
        test_model_path = get_cached_model_path(HF_BLEACHING_FILE)

        if test_model_path and os.path.isfile(test_model_path):
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            t0 = time.perf_counter()
            sess = ort.InferenceSession(
                test_model_path,
                sess_options=opts,
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
            # Run one dummy inference
            inp = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            dummy = np.zeros(shape, dtype=np.float32)
            sess.run(None, {inp.name: dummy})
            latency = (time.perf_counter() - t0) * 1000.0

            active = sess.get_providers()
            if "CoreMLExecutionProvider" in active:
                _COREML_VERIFIED = True
                _COREML_LATENCY_MS = round(latency, 2)
                return True, _COREML_LATENCY_MS
    except Exception:
        pass

    _COREML_VERIFIED = False
    _COREML_LATENCY_MS = None
    return False, None


def get_memory_budget() -> Dict[str, Any]:
    """
    Exposes both host-memory and accelerator-memory budgets.
    Explicitly distinguishes Apple unified memory: unified memory must NOT
    be budgeted as independent CPU RAM plus VRAM.
    """
    torch_info = check_torch_capabilities()
    vm = psutil.virtual_memory()

    is_unified = (sys.platform == "darwin") and (
        platform.machine().lower() in ("arm64", "aarch64") or torch_info["mps_available"]
    )

    host_total = int(vm.total)
    host_avail = int(vm.available)

    accel_total = 0
    accel_avail = 0
    accel_type = "cpu"
    accel_name = "Host CPU"

    if torch_info["cuda_available"]:
        try:
            import torch
            dev_idx = 0
            accel_total = int(torch.cuda.get_device_properties(dev_idx).total_memory)
            free_b, total_b = torch.cuda.mem_get_info(dev_idx)
            accel_avail = int(free_b)
            accel_type = "cuda"
            accel_name = torch_info["gpu_name"]
        except Exception:
            pass
    elif is_unified:
        accel_type = "apple_unified"
        accel_name = torch_info["gpu_name"]
        # Unified memory shares the host pool
        accel_total = host_total
        accel_avail = host_avail
    elif torch_info["mps_available"]:
        accel_type = "mps"
        accel_name = torch_info["gpu_name"]
        accel_total = host_total
        accel_avail = host_avail

    # Admission reserves: at least 20% of device memory or 2 GiB (whichever is larger),
    # and at least 2 GiB of available host RAM.
    min_device_reservation = max(int(0.20 * accel_total), 2 * 1024**3) if accel_total > 0 else 2 * 1024**3
    min_host_reservation = 2 * 1024**3

    return {
        "host_total_bytes": host_total,
        "host_available_bytes": host_avail,
        "host_total_gib": round(host_total / (1024**3), 2),
        "host_available_gib": round(host_avail / (1024**3), 2),
        "is_unified_memory": is_unified,
        "accelerator_type": accel_type,
        "accelerator_name": accel_name,
        "accelerator_total_bytes": accel_total,
        "accelerator_available_bytes": accel_avail,
        "accelerator_total_gib": round(accel_total / (1024**3), 2),
        "accelerator_available_gib": round(accel_avail / (1024**3), 2),
        "min_device_reservation_bytes": min_device_reservation,
        "min_host_reservation_bytes": min_host_reservation,
    }


def can_admit_worker(
    worker_peak_bytes: int = 1024**3,
    estimated_worker_vram_bytes: Optional[int] = None,
    estimated_worker_host_bytes: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Verifies memory admission for a prospective worker.
    Requires at least 20% of device memory or 2 GiB (whichever is larger) free,
    and at least 2 GiB of available host RAM.
    Supports either unified worker_peak_bytes or separate VRAM and host RAM estimates.
    """
    budget = get_memory_budget()

    host_req = estimated_worker_host_bytes if estimated_worker_host_bytes is not None else worker_peak_bytes
    vram_req = estimated_worker_vram_bytes if estimated_worker_vram_bytes is not None else worker_peak_bytes

    if budget["host_available_bytes"] < (budget["min_host_reservation_bytes"] + host_req):
        return (
            False,
            f"Insufficient host RAM: {budget['host_available_gib']} GiB available, "
            f"requires {round((budget['min_host_reservation_bytes'] + host_req) / 1024**3, 2)} GiB reserve+worker.",
        )

    if budget["accelerator_type"] == "cuda":
        total_req_vram = budget["min_device_reservation_bytes"] + vram_req
        if budget["accelerator_available_bytes"] < total_req_vram:
            return (
                False,
                f"Insufficient GPU VRAM: {budget['accelerator_available_gib']} GiB available, "
                f"requires {round(total_req_vram / 1024**3, 2)} GiB reserve+worker.",
            )

    return True, "Admission granted"


def configure_worker_threads(num_workers: int = 1, is_cuda: bool = False) -> int:
    """
    Bounds ONNX and OpenMP thread counts across workers to prevent CPU oversubscription.
    Starts CUDA inference workers with one CPU inference thread each.
    """
    cpu_cores = os.cpu_count() or 1
    if is_cuda:
        # CUDA inference workers start with 1 CPU thread each to minimize contention
        threads_per_worker = 1
    else:
        # Split CPU cores evenly across active workers, with a floor of 1
        threads_per_worker = max(1, cpu_cores // max(1, num_workers))

    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    try:
        import torch
        torch.set_num_threads(threads_per_worker)
    except Exception:
        pass

    return threads_per_worker


def get_cuda_provider_options() -> Dict[str, Any]:
    """Returns production-tuned CUDAExecutionProvider options for ONNX Runtime."""
    return {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
    }


def resolve_hardware_tier(preference: str = "auto", strict: bool = False) -> Dict[str, Any]:
    """
    Resolves hardware compute tier following priority: GPU -> Multi-Threaded CPU -> CPU.
    Separates PyTorch capabilities from ONNX Runtime providers.
    If strict=True, fails with an actionable error when requested accelerator is unavailable.
    If strict=False (or preference='auto'), records explicit CPU fallback reason.
    """
    available_ort = ort.get_available_providers()
    cpu_count = os.cpu_count() or 1
    pref = preference.lower().strip() if preference else "auto"

    torch_caps = check_torch_capabilities()
    has_cuda = torch_caps["cuda_available"]
    has_mps = torch_caps["mps_available"]

    ort_cuda_available = "CUDAExecutionProvider" in available_ort
    ort_tensorrt_available = "TensorrtExecutionProvider" in available_ort
    ort_dml_available = "DmlExecutionProvider" in available_ort
    ort_rocm_available = "ROCMExecutionProvider" in available_ort
    ort_coreml_available = "CoreMLExecutionProvider" in available_ort

    coreml_verified, coreml_latency = (False, None)
    if ort_coreml_available:
        coreml_verified, coreml_latency = verify_coreml_operational()

    gpu_available = False
    gpu_provider = "CPUExecutionProvider"
    gpu_display_name = "CPU"
    cuda_status = "unavailable"
    cuda_diagnostic = None
    cuda_fix_command = None
    cpu_fallback_reason: Optional[str] = None

    if ort_cuda_available or ort_tensorrt_available:
        gpu_available = True
        gpu_provider = "CUDAExecutionProvider" if ort_cuda_available else "TensorrtExecutionProvider"
        gpu_display_name = torch_caps["gpu_name"] if torch_caps["gpu_name"] != "CPU" else "NVIDIA CUDA GPU"
        cuda_status = "active"
    elif has_cuda:
        # PyTorch detects NVIDIA GPU with CUDA, but onnxruntime-gpu wheel is not installed
        gpu_available = True
        gpu_provider = "CPUExecutionProvider"
        gpu_display_name = f"{torch_caps['gpu_name']} (Torch Ready, ONNX Needs onnxruntime-gpu)"
        cuda_status = "onnxruntime_gpu_missing"
        cuda_diagnostic = (
            "NVIDIA CUDA GPU is detected via PyTorch, but 'onnxruntime-gpu' is not installed. "
            "ONNX models are currently falling back to CPU."
        )
        cuda_fix_command = "pip uninstall -y onnxruntime && pip install onnxruntime-gpu"
        cpu_fallback_reason = cuda_diagnostic
    elif has_mps:
        gpu_available = True
        gpu_provider = "CoreMLExecutionProvider" if coreml_verified else "CPUExecutionProvider"
        gpu_display_name = torch_caps["gpu_name"]
        if not coreml_verified:
            cpu_fallback_reason = "CoreML provider verification unverified or unsupported; ONNX models executing on CPU."
    elif ort_rocm_available:
        gpu_available = True
        gpu_provider = "ROCMExecutionProvider"
        gpu_display_name = "AMD ROCm GPU"
    elif ort_dml_available:
        gpu_available = True
        gpu_provider = "DmlExecutionProvider"
        gpu_display_name = "Windows DirectML GPU"

    # Strict accelerator enforcement
    is_strict_cuda = pref in ("cuda", "strict_cuda") or (strict and pref in ("gpu", "cuda"))
    is_strict_mps = pref in ("mps", "strict_mps") or (strict and pref in ("mps", "apple"))

    if is_strict_cuda and not (ort_cuda_available or has_cuda):
        raise RuntimeError(
            "CUDA hardware accelerator requested strictly, but no CUDA-capable device was found. "
            "Ensure NVIDIA drivers and CUDA toolkit are installed."
        )
    if is_strict_mps and not has_mps:
        raise RuntimeError(
            "Apple Silicon MPS accelerator requested strictly, but MPS is not available. "
            "This device does not support Apple Silicon Metal acceleration."
        )

    # Resolution logic
    if pref in ("gpu", "cuda", "dml", "directml", "rocm", "tensorrt", "coreml", "mps", "apple"):
        if ort_cuda_available or ort_tensorrt_available or ort_rocm_available or ort_dml_available:
            tier = "gpu"
            provider = gpu_provider
            display_name = gpu_display_name
            mode_label = f"GPU Acceleration ({gpu_display_name})"
            threads = cpu_count
        elif (has_cuda or has_mps) and not strict:
            # GPU hardware physically present; ONNX graphs run on CPU / CoreML fallback
            if coreml_verified and pref in ("gpu", "coreml", "mps", "apple"):
                tier = "gpu"
                provider = "CoreMLExecutionProvider"
                display_name = f"CoreML Accelerated [{gpu_display_name}]"
                mode_label = f"CoreML Accelerated ({gpu_display_name})"
                threads = cpu_count
            elif cpu_count > 1:
                tier = "multithread_cpu"
                provider = "CPUExecutionProvider"
                display_name = f"Multi-Threaded CPU Engine ({cpu_count} Threads) [{gpu_display_name}]"
                mode_label = f"Multi-Threaded CPU Engine (Fallback from GPU)"
                threads = cpu_count
            else:
                tier = "cpu"
                provider = "CPUExecutionProvider"
                display_name = f"Single-Threaded CPU Engine [{gpu_display_name}]"
                mode_label = "Single-Threaded CPU Engine (Fallback from GPU)"
                threads = 1
        elif strict:
            raise RuntimeError(
                f"Accelerator '{preference}' requested strictly, but no compatible ONNX execution provider is active."
            )
        else:
            # No GPU hardware
            cpu_fallback_reason = f"No GPU accelerator available; falling back from requested '{preference}' to CPU."
            if cpu_count > 1:
                tier = "multithread_cpu"
                provider = "CPUExecutionProvider"
                display_name = f"Multi-Threaded CPU Engine ({cpu_count} Threads)"
                mode_label = "Multi-Threaded CPU Engine (Fallback from GPU)"
                threads = cpu_count
            else:
                tier = "cpu"
                provider = "CPUExecutionProvider"
                display_name = "Single-Threaded CPU Engine (1 Thread)"
                mode_label = "Single-Threaded CPU Engine (Fallback from GPU)"
                threads = 1
    elif pref in (
        "multithread_cpu",
        "multithreading_cpu",
        "multithread",
        "multithreading",
        "multithreading cpu",
        "cpu_multithread",
        "parallel_cpu",
    ):
        tier = "multithread_cpu"
        provider = "CPUExecutionProvider"
        threads = cpu_count
        display_name = f"Multi-Threaded CPU Engine ({cpu_count} Threads)"
        mode_label = f"Multi-Threaded CPU Engine ({cpu_count} Threads)"
    elif pref in ("cpu", "single_cpu", "cpu_single", "singlethread", "sequential_cpu"):
        tier = "cpu"
        provider = "CPUExecutionProvider"
        threads = 1
        display_name = "Single-Threaded CPU Engine (1 Thread)"
        mode_label = "Single-Threaded CPU Engine"
    else:
        # Default Auto: GPU -> Multi-Threading CPU -> CPU
        if ort_cuda_available or ort_tensorrt_available or ort_rocm_available or ort_dml_available:
            tier = "gpu"
            provider = gpu_provider
            display_name = gpu_display_name
            mode_label = f"Auto (GPU: {gpu_display_name})"
            threads = cpu_count
        elif cpu_count > 1:
            tier = "multithread_cpu"
            provider = "CPUExecutionProvider"
            display_name = f"Multi-Threaded CPU Engine ({cpu_count} Threads)"
            if has_cuda and not ort_cuda_available:
                mode_label = f"Auto (Multi-Threaded CPU - {cpu_count} Threads) [NVIDIA GPU Needs onnxruntime-gpu]"
            elif has_mps:
                mode_label = f"Auto (Multi-Threaded CPU - {cpu_count} Threads) [Apple Silicon MPS]"
            else:
                mode_label = f"Auto (Multi-Threaded CPU - {cpu_count} Threads)"
            threads = cpu_count
        else:
            tier = "cpu"
            provider = "CPUExecutionProvider"
            display_name = "Single-Threaded CPU Engine (1 Thread)"
            mode_label = "Auto (Single-Threaded CPU)"
            threads = 1

    return {
        "tier": tier,
        "provider": provider,
        "display_name": display_name,
        "mode_label": mode_label,
        "gpu_available": gpu_available,
        "gpu_provider": gpu_provider,
        "gpu_name": gpu_display_name,
        "threads": threads,
        "cpu_count": cpu_count,
        "available_providers": available_ort,
        "torch_cuda_available": has_cuda,
        "torch_mps_available": has_mps,
        "ort_cuda_available": ort_cuda_available,
        "ort_coreml_available": ort_coreml_available,
        "coreml_verified": coreml_verified,
        "coreml_latency_ms": coreml_latency,
        "cuda_status": cuda_status,
        "cuda_diagnostic": cuda_diagnostic,
        "cuda_fix_command": cuda_fix_command,
        "cpu_fallback_reason": cpu_fallback_reason,
        "torch_capabilities": torch_caps,
    }


def get_optimal_device(preference: str = "auto") -> str:
    """Determines the best available ONNX Runtime execution provider name."""
    info = resolve_hardware_tier(preference)
    return info["provider"]


def get_device_mode(preference: str = "auto") -> str:
    """Returns resolved hardware tier mode: 'gpu', 'multithread_cpu', or 'cpu'."""
    info = resolve_hardware_tier(preference)
    return info["tier"]


def get_session_options(preference: str = "auto", thread_limit: Optional[int] = None) -> ort.SessionOptions:
    """
    Builds ONNX Runtime SessionOptions configured for the resolved hardware tier:
    - multithread_cpu: Intra/Inter-op threads bounded with ORT_PARALLEL mode.
    - cpu: 1 thread with ORT_SEQUENTIAL mode.
    - gpu: ORT_ENABLE_ALL optimizations with bounded CPU fallback threads.
    """
    info = resolve_hardware_tier(preference)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    tier = info["tier"]
    cpu_count = info["cpu_count"]
    target_threads = thread_limit if thread_limit is not None else cpu_count

    if tier == "multithread_cpu":
        opts.intra_op_num_threads = target_threads
        opts.inter_op_num_threads = min(4, target_threads)
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    elif tier == "cpu":
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:  # "gpu"
        opts.intra_op_num_threads = target_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return opts


def get_device_info(preference: str = "auto") -> Dict[str, Any]:
    """Returns comprehensive introspection details about active acceleration and memory budgets."""
    info = resolve_hardware_tier(preference)
    available = info["available_providers"]
    provider = info["provider"]
    is_gpu = info["tier"] == "gpu"
    torch_caps = info["torch_capabilities"]
    has_mps = torch_caps["mps_available"]
    has_cuda = torch_caps["cuda_available"]

    chip_label = "CPU"
    if has_mps or provider == "CoreMLExecutionProvider":
        chip_label = "Apple Silicon (MPS / CoreML)"
    elif has_cuda or provider == "CUDAExecutionProvider":
        chip_label = "NVIDIA GPU (CUDA)"

    mem_budget = get_memory_budget()

    return {
        "device": provider,
        "device_type": "gpu" if is_gpu else "cpu",
        "mode": info["tier"],
        "mode_label": info["mode_label"],
        "cuda_available": ("CUDAExecutionProvider" in available) or has_cuda,
        "torch_cuda_available": has_cuda,
        "ort_cuda_available": info["ort_cuda_available"],
        "cuda_status": info["cuda_status"],
        "cuda_diagnostic": info["cuda_diagnostic"],
        "cuda_fix_command": info["cuda_fix_command"],
        "mps_available": has_mps or ("CoreMLExecutionProvider" in available),
        "coreml_available": info["ort_coreml_available"],
        "coreml_verified": info["coreml_verified"],
        "coreml_latency_ms": info["coreml_latency_ms"],
        "dml_available": "DmlExecutionProvider" in available,
        "gpu_available": info["gpu_available"],
        "gpu_name": info["gpu_name"],
        "chip_name": chip_label,
        "compute_capability": torch_caps["compute_capability"],
        "native_precision": torch_caps["native_precision"],
        "cpu_fallback_reason": info["cpu_fallback_reason"],
        "memory_budget": mem_budget,
        "active_provider": provider,
        "threads": info["threads"],
        "cpu_count": info["cpu_count"],
        "preference": preference,
        "priority": ["gpu", "multithread_cpu", "cpu"],
    }
