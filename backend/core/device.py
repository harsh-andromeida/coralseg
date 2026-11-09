"""
Hardware device introspection and ONNX Runtime execution provider resolution.
Enforces the hardware priority hierarchy:
1. GPU (CUDA, ROCm, DirectML, TensorRT)
2. Multi-Threaded CPU (Parallel execution across all CPU cores)
3. CPU (Single-threaded sequential fallback)
"""

import os
import sys
import site
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

    # 2. Standard Linux / Colab system CUDA installations (do NOT scan generic /usr/lib)
    for p in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/bin",
        "/opt/cuda/lib64",
    ):
        if os.path.isdir(p):
            candidate_dirs.append(p)

    # Check for versioned /usr/local/cuda-* paths
    try:
        if os.path.isdir("/usr/local"):
            for entry in os.listdir("/usr/local"):
                if entry.startswith("cuda-"):
                    c_path = os.path.join("/usr/local", entry, "lib64")
                    if os.path.isdir(c_path):
                        candidate_dirs.append(c_path)
    except Exception:
        pass

    # 3. Apply discovered library paths to operating system dynamic loader
    if sys.platform.startswith("linux"):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = [p for p in candidate_dirs if p not in current_ld]
        if new_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + (f":{current_ld}" if current_ld else "")

        # Preload ONLY CUDA and cuDNN shared libraries with RTLD_GLOBAL so dlopen from ONNX Runtime resolves them.
        # NEVER preload arbitrary system libraries which exhausts glibc static TLS slots.
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


def _check_torch_accelerators() -> Tuple[bool, bool]:
    """Inspects native PyTorch GPU accelerators (CUDA and Apple Silicon MPS)."""
    has_mps = False
    has_cuda = False
    try:
        import torch
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    return has_mps, has_cuda


def get_cuda_provider_options() -> Dict[str, Any]:
    """Returns production-tuned CUDAExecutionProvider options for ONNX Runtime."""
    return {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "DEFAULT",
        "do_copy_in_default_stream": True,
    }


def resolve_hardware_tier(preference: str = "auto") -> Dict[str, Any]:
    """
    Resolves compute tier following priority: GPU -> Multi-Threaded CPU -> CPU.
    Recognizes NVIDIA CUDA, Apple Silicon GPU (MPS), ROCm, and DirectML.
    Accurately distinguishes between PyTorch CUDA availability and ONNX Runtime CUDA readiness.
    """
    available = ort.get_available_providers()
    cpu_count = os.cpu_count() or 1
    pref = preference.lower().strip() if preference else "auto"
    has_mps, has_cuda = _check_torch_accelerators()

    ort_cuda_available = "CUDAExecutionProvider" in available
    ort_tensorrt_available = "TensorrtExecutionProvider" in available
    ort_dml_available = "DmlExecutionProvider" in available
    ort_rocm_available = "ROCMExecutionProvider" in available

    gpu_available = False
    gpu_provider = "CPUExecutionProvider"
    gpu_display_name = "CPU"
    cuda_status = "unavailable"
    cuda_diagnostic = None
    cuda_fix_command = None

    if ort_cuda_available or ort_tensorrt_available:
        gpu_available = True
        gpu_provider = "CUDAExecutionProvider" if ort_cuda_available else "TensorrtExecutionProvider"
        gpu_display_name = "NVIDIA CUDA GPU"
        cuda_status = "active"
    elif has_cuda:
        # PyTorch detects NVIDIA GPU with CUDA, but onnxruntime-gpu wheel is not installed
        gpu_available = True
        gpu_provider = "CPUExecutionProvider"
        gpu_display_name = "NVIDIA CUDA GPU (Torch Ready, ONNX Needs onnxruntime-gpu)"
        cuda_status = "onnxruntime_gpu_missing"
        cuda_diagnostic = (
            "NVIDIA CUDA GPU is detected via PyTorch, but 'onnxruntime-gpu' is not installed. "
            "ONNX models (SAM, BioCLIP, YOLO) are currently falling back to CPU."
        )
        cuda_fix_command = "pip uninstall -y onnxruntime && pip install onnxruntime-gpu"
    elif has_mps:
        gpu_available = True
        gpu_provider = "CPUExecutionProvider"
        gpu_display_name = "Apple Silicon GPU (MPS)"
    elif ort_rocm_available:
        gpu_available = True
        gpu_provider = "ROCMExecutionProvider"
        gpu_display_name = "AMD ROCm GPU"
    elif ort_dml_available:
        gpu_available = True
        gpu_provider = "DmlExecutionProvider"
        gpu_display_name = "Windows DirectML GPU"

    # Explicit or automatic resolution
    if pref in ("gpu", "cuda", "dml", "directml", "rocm", "tensorrt", "coreml", "mps", "apple"):
        if ort_cuda_available or ort_tensorrt_available or ort_rocm_available or ort_dml_available:
            tier = "gpu"
            provider = gpu_provider
            display_name = gpu_display_name
            mode_label = f"GPU Acceleration ({gpu_display_name})"
            threads = cpu_count
        elif has_cuda or has_mps:
            # GPU hardware is physically present, but static ONNX graphs run on CPU fallback
            if cpu_count > 1:
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
        else:
            # No GPU hardware
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
        "available_providers": available,
        "torch_cuda_available": has_cuda,
        "ort_cuda_available": ort_cuda_available,
        "cuda_status": cuda_status,
        "cuda_diagnostic": cuda_diagnostic,
        "cuda_fix_command": cuda_fix_command,
    }


def get_optimal_device(preference: str = "auto") -> str:
    """
    Determines the best available ONNX Runtime execution provider name:
    1. GPU provider (if available in ORT)
    2. CPUExecutionProvider
    """
    info = resolve_hardware_tier(preference)
    return info["provider"]


def get_device_mode(preference: str = "auto") -> str:
    """Returns resolved hardware tier mode: 'gpu', 'multithread_cpu', or 'cpu'."""
    info = resolve_hardware_tier(preference)
    return info["tier"]


def get_session_options(preference: str = "auto") -> ort.SessionOptions:
    """
    Builds ONNX Runtime SessionOptions configured for the resolved hardware tier:
    - multithread_cpu: Intra/Inter-op threads set to all CPU cores with ORT_PARALLEL mode.
    - cpu: 1 thread with ORT_SEQUENTIAL mode.
    - gpu: ORT_ENABLE_ALL optimizations with CPU fallback threads.
    """
    info = resolve_hardware_tier(preference)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    tier = info["tier"]
    cpu_count = info["cpu_count"]

    if tier == "multithread_cpu":
        opts.intra_op_num_threads = cpu_count
        opts.inter_op_num_threads = min(4, cpu_count)
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    elif tier == "cpu":
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:  # "gpu"
        opts.intra_op_num_threads = cpu_count
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return opts


def get_device_info(preference: str = "auto") -> Dict[str, Any]:
    """Returns comprehensive introspection details about active acceleration."""
    info = resolve_hardware_tier(preference)
    available = info["available_providers"]
    provider = info["provider"]
    is_gpu = info["tier"] == "gpu"
    has_mps, has_cuda = _check_torch_accelerators()

    chip_label = "CPU"
    if has_mps or provider == "CoreMLExecutionProvider":
        chip_label = "Apple Silicon (MPS / CoreML)"
    elif has_cuda or provider == "CUDAExecutionProvider":
        chip_label = "NVIDIA GPU (CUDA)"

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
        "coreml_available": "CoreMLExecutionProvider" in available,
        "dml_available": "DmlExecutionProvider" in available,
        "gpu_available": info["gpu_available"],
        "gpu_name": info["gpu_name"],
        "chip_name": chip_label,
        "active_provider": provider,
        "threads": info["threads"],
        "cpu_count": info["cpu_count"],
        "preference": preference,
        "priority": ["gpu", "multithread_cpu", "cpu"],
    }
