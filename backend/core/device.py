"""
Hardware device introspection and ONNX Runtime execution provider resolution.
Enforces the hardware priority hierarchy:
1. GPU (CUDA, ROCm, DirectML, TensorRT)
2. Multi-Threaded CPU (Parallel execution across all CPU cores)
3. CPU (Single-threaded sequential fallback)
"""

import os
from typing import Dict, Any, Tuple
import onnxruntime as ort

GPU_PROVIDERS = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "TensorrtExecutionProvider",
]


def resolve_hardware_tier(preference: str = "auto") -> Dict[str, Any]:
    """
    Resolves compute tier following priority: GPU -> Multi-Threaded CPU -> CPU.
    """
    available = ort.get_available_providers()
    cpu_count = os.cpu_count() or 1
    pref = preference.lower().strip() if preference else "auto"

    gpu_provider = next((p for p in GPU_PROVIDERS if p in available), None)
    gpu_available = gpu_provider is not None

    if gpu_provider == "CUDAExecutionProvider":
        gpu_display_name = "NVIDIA CUDA GPU"
    elif gpu_provider == "ROCMExecutionProvider":
        gpu_display_name = "AMD ROCm GPU"
    elif gpu_provider == "DmlExecutionProvider":
        gpu_display_name = "Windows DirectML GPU"
    elif gpu_provider == "TensorrtExecutionProvider":
        gpu_display_name = "NVIDIA TensorRT GPU"
    else:
        gpu_display_name = "GPU Accelerator"

    # Explicit or automatic resolution
    if pref in ("gpu", "cuda", "dml", "directml", "rocm", "tensorrt"):
        if gpu_available:
            tier = "gpu"
            provider = gpu_provider
            display_name = gpu_display_name
            mode_label = f"GPU Acceleration ({gpu_display_name})"
            threads = cpu_count
        else:
            # Fallback down the hierarchy: multithreading cpu -> cpu
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
        if gpu_available:
            tier = "gpu"
            provider = gpu_provider
            display_name = gpu_display_name
            mode_label = f"Auto (GPU: {gpu_display_name})"
            threads = cpu_count
        elif cpu_count > 1:
            tier = "multithread_cpu"
            provider = "CPUExecutionProvider"
            display_name = f"Multi-Threaded CPU Engine ({cpu_count} Threads)"
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
        "gpu_name": display_name,
        "threads": threads,
        "cpu_count": cpu_count,
        "available_providers": available,
    }


def get_optimal_device(preference: str = "auto") -> str:
    """
    Determines the best available ONNX Runtime execution provider name:
    1. GPU provider (if available)
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

    return {
        "device": provider,
        "device_type": "gpu" if is_gpu else "cpu",
        "mode": info["tier"],
        "mode_label": info["mode_label"],
        "cuda_available": "CUDAExecutionProvider" in available,
        "mps_available": False,
        "dml_available": "DmlExecutionProvider" in available,
        "gpu_available": info["gpu_available"],
        "gpu_name": info["display_name"],
        "chip_name": "CPU",
        "active_provider": provider,
        "threads": info["threads"],
        "cpu_count": info["cpu_count"],
        "preference": preference,
        "priority": ["gpu", "multithread_cpu", "cpu"],
    }

