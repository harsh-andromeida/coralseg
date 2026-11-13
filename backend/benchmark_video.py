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

import cv2
from core.device import check_torch_capabilities, get_memory_budget, get_optimal_device
from core.models import (
    load_coralscop_model,
    load_bioclip_model,
    load_bleaching_model,
)
from core.video.engine import VideoProcessor, VideoJobState
from core.video.sam2_tracker import SAM2BenthicTracker


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
    """Measures current, peak allocated, and total system GPU memory (including ONNX/driver) in MiB."""
    res = {"allocated_mib": 0.0, "peak_mib": 0.0, "total_used_mib": 0.0, "device_total_mib": 0.0}
    if not _HAS_TORCH:
        return res

    if device.startswith("cuda") and torch.cuda.is_available():
        res["allocated_mib"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        res["peak_mib"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            res["total_used_mib"] = round((total_bytes - free_bytes) / (1024 * 1024), 2)
            res["device_total_mib"] = round(total_bytes / (1024 * 1024), 2)
        except Exception:
            pass
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
    compile_tracker: bool = False,
    export_mp4: bool = False,
    trial_index: int = 1,
) -> Dict[str, Any]:
    """Executes a single benchmark trial and collects fine-grained telemetry."""
    print(f"\n[Trial {trial_index}] Starting benchmark: profile={model_profile}, size={model_size}, workers={worker_count}, device={device}, precision={precision}, compiled={compile_tracker}...")

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "exports", "benchmark_cache")
    os.makedirs(cache_dir, exist_ok=True)

    processor = VideoProcessor(
        sam_loader=lambda: load_coralscop_model(device=device),
        bioclip_loader=lambda: load_bioclip_model(device=device),
        bleaching_loader=lambda: load_bleaching_model(device=device),
        cache_dir=cache_dir,
    )

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
            "compile_tracker": compile_tracker,
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
    timings = completed_job.get("timings") or {}
    export_paths = completed_job.get("export_paths") or {}

    # Output deliverable verification
    export_verification = {
        "csv_exists": False,
        "csv_rows": 0,
        "coco_exists": False,
        "coco_annotations": 0,
        "mp4_exists": False,
        "mp4_frames": 0,
    }
    if "csv" in export_paths and os.path.isfile(export_paths["csv"]):
        export_verification["csv_exists"] = True
        try:
            with open(export_paths["csv"], "r", encoding="utf-8") as f:
                export_verification["csv_rows"] = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    if "coco_json" in export_paths and os.path.isfile(export_paths["coco_json"]):
        export_verification["coco_exists"] = True
        try:
            with open(export_paths["coco_json"], "r", encoding="utf-8") as f:
                coco_data = json.load(f)
                export_verification["coco_annotations"] = len(coco_data.get("annotations", []))
        except Exception:
            pass
    if "annotated_mp4" in export_paths and os.path.isfile(export_paths["annotated_mp4"]):
        export_verification["mp4_exists"] = True
        try:
            cap_v = cv2.VideoCapture(export_paths["annotated_mp4"])
            export_verification["mp4_frames"] = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_v.release()
        except Exception:
            pass

    t_compile = timings.get("compile_warmup_sec", 0.0)
    t_init = timings.get("initialization_sec", 0.0)
    t_gpu = timings.get("synchronized_gpu_inference_sec", 0.0)
    t_disc = timings.get("keyframe_discovery_sec", 0.0)
    t_bg_class = timings.get("background_classification_sec", 0.0)
    t_prep = timings.get("video_decode_prep_sec", 0.0)
    t_geom = timings.get("geometry_extraction_sec", 0.0) + timings.get("mask_transfer_sec", 0.0)
    t_queue = timings.get("queue_wait_sec", 0.0)
    t_pers = timings.get("persistence_sec", 0.0)
    t_deliv = timings.get("export_deliverables_sec", 0.0)
    t_mp4 = timings.get("export_annotated_mp4_sec", 0.0)
    t_idle = timings.get("unaccounted_idle_sec", 0.0)
    t_crit = timings.get("critical_path_elapsed_sec", wall_duration)

    print(f"\n[Trial {trial_index}] Done in {wall_duration}s | Source FPS: {source_fps} | Tracker FPS: {tracker_fps} | Corals: {unique_corals} | Peak RSS: {peak_rss} MiB")
    if accel_mem.get("total_used_mib", 0.0) > 0:
        print(f"  GPU VRAM: {accel_mem['allocated_mib']} MiB allocated, {accel_mem['peak_mib']} MiB peak (Total Used on Device: {accel_mem['total_used_mib']} / {accel_mem['device_total_mib']} MiB)")
    print(f"  Verification: CSV rows={export_verification['csv_rows']}, COCO annotations={export_verification['coco_annotations']}, MP4 frames={export_verification['mp4_frames']}")

    if timings:
        print("=" * 96)
        print(f"Pipeline Stage Breakdown for {total_frames} frames (Wall Duration: {wall_duration}s | {source_fps} FPS)")
        print("=" * 96)
        print(f"{'Critical-Path Sequential Stages':<52} | {'Duration (s)':<12} | {'% Wall Time':<12} | {'Effective Rate'}")
        print("-" * 96)
        print(f"{'1. Video Decoding & Chunk Frame Buffer (in-memory)':<52} | {t_prep:>10.2f} s | {(t_prep/max(wall_duration, 0.001))*100:>10.1f} % | {total_frames/max(t_prep, 0.001):>8.1f} FPS")
        print(f"{'2. Keyframe Discovery (CoralSCOP ViT-B)':<52} | {t_disc:>10.2f} s | {(t_disc/max(wall_duration, 0.001))*100:>10.1f} % | {keyframes_count/max(t_disc, 0.001):>8.1f} kf/s")
        print(f"{'3. Synchronized GPU Inference (SAM/TAM)':<52} | {t_gpu:>10.2f} s | {(t_gpu/max(wall_duration, 0.001))*100:>10.1f} % | {total_frames/max(t_gpu, 0.001):>8.1f} FPS")
        print(f"{'4. Mask Transfer & Geometry Extraction':<52} | {t_geom:>10.2f} s | {(t_geom/max(wall_duration, 0.001))*100:>10.1f} % | {total_frames/max(t_geom, 0.001):>8.1f} FPS")
        print(f"{'5. SQLite Persistence & Progress Updates':<52} | {t_pers:>10.2f} s | {(t_pers/max(wall_duration, 0.001))*100:>10.1f} % | {total_frames/max(t_pers, 0.001):>8.1f} FPS")
        print(f"{'6. Deliverables Export (CSV + COCO JSON)':<52} | {t_deliv:>10.2f} s | {(t_deliv/max(wall_duration, 0.001))*100:>10.1f} % | -")
        if export_mp4:
            print(f"{'7. Annotated MP4 Video Export':<52} | {t_mp4:>10.2f} s | {(t_mp4/max(wall_duration, 0.001))*100:>10.1f} % | {total_frames/max(t_mp4, 0.001):>8.1f} FPS")
        print(f"{'8. Unaccounted Gaps / Overhead':<52} | {t_idle:>10.2f} s | {(t_idle/max(wall_duration, 0.001))*100:>10.1f} % | -")
        print("-" * 96)
        print(f"{'Background Asynchronous Tasks (Concurrent)':<52} | {'Duration (s)':<12} | {'Status':<12} | {'Overlap Note'}")
        print("-" * 96)
        print(f"{'• Keyframe Classification (BioCLIP + Bleach)':<52} | {t_bg_class:>10.2f} s | {'Non-blocking':<12} | Overlaps tracking loop")
        print(f"{'• Inference Queue Wait (Worker Idle)':<52} | {t_queue:>10.2f} s | {'Bounded':<12} | CPU prep overlap")
        print("-" * 96)
        print(f"{'Cold Startup & Graph Compilation':<52} | {'Duration (s)':<12} | {'Note':<12} | {'Impact'}")
        print("-" * 96)
        print(f"{'• Ahead-of-Time Graph Compilation & Warmup':<52} | {t_compile:>10.2f} s | {'AOT Warmup':<12} | Excluded from warm FPS")
        print(f"{'• Model Load & Weights Initialization':<52} | {t_init:>10.2f} s | {'One-time':<12} | Reused across jobs")
        print("=" * 96)

    return {
        "trial_index": trial_index,
        "is_cold": (trial_index == 1),
        "wall_duration_sec": wall_duration,
        "critical_path_elapsed_sec": t_crit,
        "compile_warmup_sec": t_compile,
        "total_frames": total_frames,
        "source_fps": source_fps,
        "tracker_fps": tracker_fps,
        "keyframes_count": keyframes_count,
        "unique_corals": unique_corals,
        "peak_rss_mib": peak_rss,
        "accelerator_allocated_mib": accel_mem.get("allocated_mib", 0.0),
        "accelerator_peak_mib": accel_mem.get("peak_mib", 0.0),
        "accelerator_total_used_mib": accel_mem.get("total_used_mib", 0.0),
        "export_paths": completed_job.get("export_paths", {}),
        "export_verification": export_verification,
        "timings": timings,
    }


def run_fixed_object_benchmark(
    video_path: str,
    object_counts: List[int] = [1, 8, 16],
    frame_count: int = 100,
    model_profile: str = "sam2",
    model_size: int = 512,
    device: str = "auto",
    precision: Optional[str] = None,
    compile_tracker: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates pure SAM2/TAM propagation scalability under fixed object counts (1, 8, 16).
    Isolates neural inference from detection/classification to establish scaling characteristics.
    """
    print("\n" + "=" * 70)
    print(f"Fixed Object Propagation Benchmark ({object_counts} colonies across {frame_count} frames)")
    print(f"Configuration: profile={model_profile}, size={model_size}, device={device}, compiled={compile_tracker}")
    print("=" * 70)

    # 1. Load video frames
    cap = cv2.VideoCapture(video_path)
    frames_rgb: List[np.ndarray] = []
    while len(frames_rgb) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
    cap.release()

    if not frames_rgb:
        raise RuntimeError(f"Could not read frames from {video_path}")

    actual_frames = len(frames_rgb)
    orig_h, orig_w = frames_rgb[0].shape[:2]

    # 2. Build Tracker
    tracker = SAM2BenthicTracker(
        device=device,
        model_size=model_size,
        precision=precision,
        compile_tracker=compile_tracker,
        offload_to_cpu=True,
    )
    tracker.load()

    results: List[Dict[str, Any]] = []

    for obj_cnt in object_counts:
        print(f"\nEvaluating fixed {obj_cnt} colony tracking across {actual_frames} frames...")
        tracker.init_video(frames_rgb, frame_offset=0, orig_hw=(orig_h, orig_w))

        # Register synthetic colony masks on frame 0
        grid_cols = int(np.ceil(np.sqrt(obj_cnt)))
        pad_x = orig_w // (grid_cols + 1)
        pad_y = orig_h // (grid_cols + 1)
        box_radius = min(pad_x, pad_y) // 3

        registered = 0
        for idx in range(obj_cnt):
            row = idx // grid_cols
            col = idx % grid_cols
            cx = pad_x * (col + 1)
            cy = pad_y * (row + 1)

            mask = np.zeros((orig_h, orig_w), dtype=bool)
            y1, y2 = max(0, cy - box_radius), min(orig_h, cy + box_radius)
            x1, x2 = max(0, cx - box_radius), min(orig_w, cx + box_radius)
            mask[y1:y2, x1:x2] = True

            meta = {"genus": f"Colony_{idx+1}", "growth_form": "Massive", "condition": "Healthy"}
            tracker.register_mask(frame_idx=0, obj_id=idx + 1, binary_mask=mask, metadata=meta)
            registered += 1

        # Synchronize before propagation timing
        if tracker.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        t_prop_start = time.perf_counter()
        propagated_frames = 0
        total_tracked_masks = 0

        for out_f, det_tracks in tracker.propagate(start_frame_idx=1, end_frame_idx=actual_frames):
            propagated_frames += 1
            total_tracked_masks += len(det_tracks)

        if tracker.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        prop_duration = max(0.001, time.perf_counter() - t_prop_start)
        fps = round(propagated_frames / prop_duration, 1)
        mem = get_accelerator_memory_mb(tracker.device)
        timings = tracker.get_timings()

        print(f"  -> {obj_cnt} colonies: {actual_frames} frames in {prop_duration:.2f}s | {fps} FPS | Avg masks/frame: {total_tracked_masks/max(propagated_frames, 1):.1f}")
        results.append({
            "object_count": obj_cnt,
            "frames_tested": propagated_frames,
            "propagation_sec": round(prop_duration, 3),
            "propagation_fps": fps,
            "avg_masks_per_frame": round(total_tracked_masks / max(propagated_frames, 1), 2),
            "gpu_allocated_mib": mem.get("allocated_mib", 0.0),
            "gpu_peak_mib": mem.get("peak_mib", 0.0),
            "gpu_total_used_mib": mem.get("total_used_mib", 0.0),
            "tracker_timings": timings,
        })
        tracker.reset()

    tracker.close()
    return {"configuration": {"model_size": model_size, "device": device, "compiled": compile_tracker}, "results": results}


def main():
    parser = argparse.ArgumentParser(description="ReefVision Video Pipeline Benchmark Harness")
    parser.add_argument("--video", type=str, default=None, help="Path to input video (.mp4)")
    parser.add_argument("--model-profile", type=str, default="sam2", choices=["sam2", "efficienttam"], help="Model profile")
    parser.add_argument("--model-size", type=int, default=512, choices=[512, 1024], help="Model input size")
    parser.add_argument("--worker-count", type=int, default=1, help="Number of parallel inference workers")
    parser.add_argument("--device", type=str, default="auto", help="Device preference (cuda, mps, cpu, auto)")
    parser.add_argument("--precision", type=str, default=None, choices=[None, "fp16", "bf16", "fp32"], help="Floating point precision")
    parser.add_argument("--compile-tracker", action="store_true", help="Enable ahead-of-time graph compilation (CUDA vos_optimized)")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeated trials")
    parser.add_argument("--fixed-objects", action="store_true", help="Also run fixed 1, 8, 16 colony tracking benchmark")
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
    print(f"Model Configuration: {args.model_profile} @ {args.model_size}px (Workers: {args.worker_count}, Compiled: {args.compile_tracker})")
    print(f"Video Source: {os.path.basename(video_path)}")
    print("=" * 70)

    # 3. Optional Fixed Object Benchmark
    fixed_obj_results = None
    if args.fixed_objects:
        fixed_obj_results = run_fixed_object_benchmark(
            video_path=video_path,
            object_counts=[1, 8, 16],
            frame_count=100,
            model_profile=args.model_profile,
            model_size=args.model_size,
            device=args.device,
            precision=args.precision,
            compile_tracker=args.compile_tracker,
        )

    # 4. Execute Full Pipeline Trials
    trial_results = []
    for i in range(args.trials):
        res = run_benchmark_trial(
            video_path=video_path,
            model_profile=args.model_profile,
            model_size=args.model_size,
            device=args.device,
            precision=args.precision,
            worker_count=args.worker_count,
            compile_tracker=args.compile_tracker,
            export_mp4=args.export_mp4,
            trial_index=i + 1,
        )
        trial_results.append(res)

    # 5. Compute Aggregate Statistics (Separate Cold from Warm)
    warm_trials = trial_results[1:] if len(trial_results) > 1 else trial_results
    warm_source_fps = [r["source_fps"] for r in warm_trials]
    warm_tracker_fps = [r["tracker_fps"] for r in warm_trials]
    warm_durations = [r["wall_duration_sec"] for r in warm_trials]
    peak_rss = max(r["peak_rss_mib"] for r in trial_results)
    max_accel_peak = max(r["accelerator_peak_mib"] for r in trial_results)
    max_accel_used = max(r["accelerator_total_used_mib"] for r in trial_results)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "configuration": {
            "model_profile": args.model_profile,
            "model_size": args.model_size,
            "worker_count": args.worker_count,
            "compile_tracker": args.compile_tracker,
            "device": args.device,
            "precision": args.precision,
            "video_filename": os.path.basename(video_path),
            "export_mp4": args.export_mp4,
        },
        "metrics": {
            "cold_trial": trial_results[0],
            "warm_trials_count": len(warm_trials),
            "warm_mean_source_fps": round(float(np.mean(warm_source_fps)), 2),
            "warm_std_source_fps": round(float(np.std(warm_source_fps)), 2) if len(warm_trials) > 1 else 0.0,
            "warm_mean_tracker_fps": round(float(np.mean(warm_tracker_fps)), 2),
            "warm_mean_duration_sec": round(float(np.mean(warm_durations)), 2),
            "peak_rss_mib": peak_rss,
            "accelerator_peak_mib": max_accel_peak,
            "accelerator_total_used_mib": max_accel_used,
        },
        "trials": trial_results,
        "fixed_object_benchmark": fixed_obj_results,
    }

    # 6. Output JSON
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nStructured benchmark results saved to: {args.output_json}")

    # 7. Output Markdown Summary Table & Latency Breakdown
    latest_timings = trial_results[-1].get("timings", {}) if trial_results else {}
    breakdown_section = ""
    if latest_timings:
        t_compile = latest_timings.get("compile_warmup_sec", 0.0)
        t_gpu = latest_timings.get("synchronized_gpu_inference_sec", 0.0)
        t_disc = latest_timings.get("keyframe_discovery_sec", 0.0)
        t_bg_class = latest_timings.get("background_classification_sec", 0.0)
        t_prep = latest_timings.get("video_decode_prep_sec", 0.0)
        t_geom = latest_timings.get("geometry_extraction_sec", 0.0) + latest_timings.get("mask_transfer_sec", 0.0)
        t_queue = latest_timings.get("queue_wait_sec", 0.0)
        t_pers = latest_timings.get("persistence_sec", 0.0)
        t_deliv = latest_timings.get("export_deliverables_sec", 0.0)
        t_mp4 = latest_timings.get("export_annotated_mp4_sec", 0.0)
        t_idle = latest_timings.get("unaccounted_idle_sec", 0.0)
        t_wall = trial_results[-1].get("wall_duration_sec", 0.001)
        tot_f = trial_results[-1].get("total_frames", 202)
        v_info = trial_results[-1].get("export_verification", {})

        breakdown_section = f"""
### Per-Stage Latency Breakdown ({tot_f} frames, {t_wall:.2f}s total)

| Pipeline Stage | Wall Time (s) | Share (%) | Effective Stage Rate | Execution Mode |
|---|---:|---:|---:|---|
| **1. Video Decoding & Chunk Frame Prep** | {t_prep:.2f}s | {(t_prep/t_wall)*100:.1f}% | {tot_f/max(t_prep, 0.001):.1f} FPS | In-memory Bounded Buffer |
| **2. Keyframe Discovery (CoralSCOP ViT-B)** | {t_disc:.2f}s | {(t_disc/t_wall)*100:.1f}% | - | Sequential Critical-Path |
| **3. Synchronized GPU Inference (SAM/TAM)** | {t_gpu:.2f}s | {(t_gpu/t_wall)*100:.1f}% | {tot_f/max(t_gpu, 0.001):.1f} FPS | {'Compiled AOT' if args.compile_tracker else 'Eager'} |
| **4. Mask CPU Transfer & Geometry** | {t_geom:.2f}s | {(t_geom/t_wall)*100:.1f}% | {tot_f/max(t_geom, 0.001):.1f} FPS | Downstream Overlap |
| **5. SQLite Persistence & Batched Writes** | {t_pers:.2f}s | {(t_pers/t_wall)*100:.1f}% | {tot_f/max(t_pers, 0.001):.1f} FPS | Critical-Path Batched |
| **6. Deliverables Export (CSV + COCO JSON)** | {t_deliv:.2f}s | {(t_deliv/t_wall)*100:.1f}% | - | Post-Processing |
| **7. Annotated MP4 Video Export** | {t_mp4:.2f}s | {(t_mp4/t_wall)*100:.1f}% | {tot_f/max(t_mp4, 0.001):.1f} FPS | Single-Pass Encoder |
| **8. Unaccounted Gaps / Overhead** | {t_idle:.2f}s | {(t_idle/t_wall)*100:.1f}% | - | Coordination Overhead |

#### Asynchronous & Ahead-of-Time Stages
- **Background Classification (BioCLIP + Bleach)**: {t_bg_class:.2f}s (overlapped concurrently across worker thread pool)
- **Inference Queue Wait**: {t_queue:.2f}s (bounded prefetch overlap)
- **Compilation Warmup**: {t_compile:.2f}s (ahead-of-time graph compilation on trial 1)

#### Deliverable & Output Verification
- **CSV Transect Summary**: {v_info.get('csv_rows', 0)} rows (verified: {v_info.get('csv_exists', False)})
- **COCO Video JSON**: {v_info.get('coco_annotations', 0)} polygon annotations (verified: {v_info.get('coco_exists', False)})
- **Annotated MP4**: {v_info.get('mp4_frames', 0)} frames (verified: {v_info.get('mp4_exists', False)})
"""

    fixed_table = ""
    if fixed_obj_results:
        fixed_rows = "\n".join(
            f"| {r['object_count']} | {r['frames_tested']} | {r['propagation_sec']}s | **{r['propagation_fps']} FPS** | {r['gpu_peak_mib']} MiB |"
            for r in fixed_obj_results["results"]
        )
        fixed_table = f"""
### Fixed Object Scaling Benchmark (Pure Tracker Propagation)

| Colony Count | Frames | Duration (s) | Propagation Rate | Peak Accelerator Memory |
|---|---:|---:|---:|---:|
{fixed_rows}
"""

    md_table = f"""
# ReefVision Video Pipeline Benchmark Report

**Date**: {summary['timestamp']}  
**Hardware**: {device_name} ({torch_device})  
**Unified Memory**: {is_unified} | **PyTorch**: {torch_version}  

| Configuration | Model Size | Workers | Compiled | Warm Source FPS | Warm Tracker FPS | Duration (s) | Peak Host RAM | Peak GPU VRAM |
|---|---|---:|---|---:|---:|---:|---:|---:|
| {args.model_profile.upper()} ({args.device}) | {args.model_size} | {args.worker_count} | {args.compile_tracker} | **{summary['metrics']['warm_mean_source_fps']}** | {summary['metrics']['warm_mean_tracker_fps']} | {summary['metrics']['warm_mean_duration_sec']}s | {peak_rss} MiB | {max_accel_peak} MiB |

{breakdown_section}
{fixed_table}
"""
    print(md_table)

    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(md_table)
        print(f"Markdown report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
