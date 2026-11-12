"""
Benchmark Harness for ReefVision Video Pipeline on CUDA and Apple Silicon (M5/MPS).
Measures source FPS, end-to-end throughput, per-stage latency breakdown,
peak host RAM (RSS), and peak accelerator memory across worker counts and model profiles.
Outputs structured JSON results and a formatted Markdown report table.
"""

import os
import sys
import time
import json
import argparse
import platform
import resource
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from core.device import check_torch_capabilities, get_memory_budget, get_optimal_device
from core.models import (
    load_coralscop_model,
    load_bioclip_model,
    load_bleaching_model,
)
from core.video.engine import VideoProcessor, VideoJobState


def get_peak_rss_mb() -> float:
    """Returns peak resident set size (RSS) in MiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        # macOS reports in bytes
        return round(usage / (1024 * 1024), 2)
    else:
        # Linux reports in KiB
        return round(usage / 1024, 2)


def get_accelerator_memory_mb(device: str) -> Dict[str, float]:
    """Measures current and peak allocated accelerator memory in MiB."""
    res = {"allocated_mib": 0.0, "peak_mib": 0.0}
    if not _HAS_TORCH:
        return res

    if device.startswith("cuda") and torch.cuda.is_available():
        res["allocated_mib"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        res["peak_mib"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    elif device.startswith("mps") and hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                res["allocated_mib"] = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
            if hasattr(torch.mps, "driver_allocated_memory"):
                res["peak_mib"] = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
        except Exception:
            pass
    return res


def run_benchmark_trial(
    video_path: str,
    model_profile: str = "sam2",
    model_size: int = 512,
    device: str = "auto",
    precision: Optional[str] = None,
    worker_count: int = 1,
    export_mp4: bool = False,
    trial_index: int = 1,
) -> Dict[str, Any]:
    """Executes a single benchmark trial and collects fine-grained telemetry."""
    print(f"\n[Trial {trial_index}] Starting benchmark: profile={model_profile}, size={model_size}, workers={worker_count}, device={device}, precision={precision}...")

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "exports", "benchmark_cache")
    os.makedirs(cache_dir, exist_ok=True)

    processor = VideoProcessor(
        sam_loader=lambda: load_coralscop_model(device=device),
        bioclip_loader=lambda: load_bioclip_model(device=device),
        bleaching_loader=lambda: load_bleaching_model(device=device),
        cache_dir=cache_dir,
    )

    start_rss = get_peak_rss_mb()
    t0 = time.perf_counter()

    job_id = processor.start_job(
        video_path=video_path,
        params={
            "overlap_threshold": 0.70,
            "min_stride": 15,
            "max_stride": 45,
            "points_per_side": 8,
            "tracker_mode": model_profile,
            "model_profile": model_profile,
            "model_size": model_size,
            "device_pref": device,
            "precision": precision,
            "worker_count": worker_count,
            "export_mp4": export_mp4,
        },
    )

    # Monitor until completion
    completed_job = None
    poll_count = 0
    while True:
        time.sleep(0.5)
        poll_count += 1
        job = processor.get_job(job_id)
        if not job:
            continue

        state = job.get("state")
        if poll_count % 10 == 0:
            print(f"  Frame {job.get('current_frame', 0)}/{job.get('total_frames', 0)} ({job.get('progress_pct', 0.0):.1f}%) | {job.get('analysis_fps', 0.0):.1f} FPS | state={state}")

        if state in (VideoJobState.COMPLETED, VideoJobState.ERROR, VideoJobState.CANCELLED):
            completed_job = job
            break

    wall_duration = round(time.perf_counter() - t0, 3)
    peak_rss = get_peak_rss_mb()
    accel_mem = get_accelerator_memory_mb(device)

    processor.close()

    if not completed_job or completed_job.get("state") != VideoJobState.COMPLETED:
        err = completed_job.get("error") if completed_job else "Job failed or timed out"
        raise RuntimeError(f"Trial {trial_index} failed: {err}")

    total_frames = completed_job.get("total_frames", 0)
    source_fps = round(total_frames / max(wall_duration, 0.001), 2)
    tracker_fps = completed_job.get("tracker_fps", 0.0)
    keyframes_count = completed_job.get("keyframes_count", 0)
    unique_corals = completed_job.get("total_unique_corals", 0)

    print(f"[Trial {trial_index}] Done in {wall_duration}s | Source FPS: {source_fps} | Tracker FPS: {tracker_fps} | Corals: {unique_corals} | Peak RSS: {peak_rss} MiB")

    return {
        "trial_index": trial_index,
        "wall_duration_sec": wall_duration,
        "total_frames": total_frames,
        "source_fps": source_fps,
        "tracker_fps": tracker_fps,
        "keyframes_count": keyframes_count,
        "unique_corals": unique_corals,
        "peak_rss_mib": peak_rss,
        "accelerator_allocated_mib": accel_mem.get("allocated_mib", 0.0),
        "accelerator_peak_mib": accel_mem.get("peak_mib", 0.0),
        "export_paths": completed_job.get("export_paths", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="ReefVision Video Pipeline Benchmark Harness")
    parser.add_argument("--video", type=str, default=None, help="Path to input video (.mp4)")
    parser.add_argument("--model-profile", type=str, default="sam2", choices=["sam2", "efficienttam"], help="Model profile")
    parser.add_argument("--model-size", type=int, default=512, choices=[512, 1024], help="Model input size")
    parser.add_argument("--worker-count", type=int, default=1, help="Number of parallel inference workers")
    parser.add_argument("--device", type=str, default="auto", help="Device preference (cuda, mps, cpu, auto)")
    parser.add_argument("--precision", type=str, default=None, choices=[None, "fp16", "bf16", "fp32"], help="Floating point precision")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeated trials")
    parser.add_argument("--export-mp4", action="store_true", help="Include annotated MP4 rendering in benchmark")
    parser.add_argument("--output-json", type=str, default=None, help="Save structured results to JSON file")
    parser.add_argument("--output-report", type=str, default=None, help="Save markdown comparison report")
    args = parser.parse_args()

    # 1. Resolve video path
    video_path = args.video
    if not video_path:
        default_sample = os.path.join(os.path.dirname(__file__), "..", "demo_images", "sample_reef_transect_25fps.mp4")
        if os.path.isfile(default_sample):
            video_path = default_sample
        else:
            raise FileNotFoundError("No video provided and demo_images/sample_reef_transect_25fps.mp4 not found.")

    # 2. Inspect Hardware
    hw_info = check_torch_capabilities()
    budget = get_memory_budget()

    torch_version = "N/A"
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        pass

    device_name = hw_info.get("gpu_name") or "CPU"
    torch_device = "mps" if hw_info.get("mps_available") else ("cuda" if hw_info.get("cuda_available") else "cpu")
    is_unified = budget.get("is_unified_memory", False)

    print("=" * 70)
    print("ReefVision SAM Video Optimization Benchmark Harness")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.machine()} ({platform.processor()})")
    print(f"PyTorch Version: {torch_version}")
    print(f"Torch Device: {torch_device} (Device Name: {device_name})")
    print(f"Unified Memory: {is_unified}")
    print(f"Host Available RAM: {budget.get('host_available_bytes', 0) / (1024**3):.2f} GiB")
    print(f"Model Configuration: {args.model_profile} @ {args.model_size}px (Workers: {args.worker_count})")
    print(f"Video Source: {os.path.basename(video_path)}")
    print("=" * 70)

    # 3. Execute Trials
    trial_results = []
    for i in range(args.trials):
        res = run_benchmark_trial(
            video_path=video_path,
            model_profile=args.model_profile,
            model_size=args.model_size,
            device=args.device,
            precision=args.precision,
            worker_count=args.worker_count,
            export_mp4=args.export_mp4,
            trial_index=i + 1,
        )
        trial_results.append(res)

    # 4. Compute Aggregate Statistics
    source_fps_list = [r["source_fps"] for r in trial_results]
    tracker_fps_list = [r["tracker_fps"] for r in trial_results]
    durations = [r["wall_duration_sec"] for r in trial_results]
    peak_rss = max(r["peak_rss_mib"] for r in trial_results)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "configuration": {
            "model_profile": args.model_profile,
            "model_size": args.model_size,
            "worker_count": args.worker_count,
            "device": args.device,
            "precision": args.precision,
            "video_filename": os.path.basename(video_path),
            "export_mp4": args.export_mp4,
        },
        "metrics": {
            "mean_source_fps": round(float(np.mean(source_fps_list)), 2),
            "std_source_fps": round(float(np.std(source_fps_list)), 2),
            "mean_tracker_fps": round(float(np.mean(tracker_fps_list)), 2),
            "mean_duration_sec": round(float(np.mean(durations)), 2),
            "peak_rss_mib": peak_rss,
        },
        "trials": trial_results,
    }

    # 5. Output JSON
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nStructured benchmark results saved to: {args.output_json}")

    # 6. Output Markdown Summary Table
    md_table = f"""
# ReefVision Video Pipeline Benchmark Report

**Date**: {summary['timestamp']}  
**Hardware**: {device_name} ({torch_device})  
**Unified Memory**: {is_unified} | **PyTorch**: {torch_version}  

| Configuration | Model Size | Workers | Mean Source FPS | Mean Tracker FPS | Duration (s) | Peak Host RAM |
|---|---|---:|---:|---:|---:|---:|
| {args.model_profile.upper()} ({args.device}) | {args.model_size} | {args.worker_count} | **{summary['metrics']['mean_source_fps']}** | {summary['metrics']['mean_tracker_fps']} | {summary['metrics']['mean_duration_sec']}s | {peak_rss} MiB |

*Target threshold: 10 FPS minimum, 15-20 FPS production target.*
"""
    print(md_table)

    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(md_table)
        print(f"Markdown report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
