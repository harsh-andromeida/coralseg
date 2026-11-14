"""
App-flow benchmark for the ReefVision video pipeline.

Drives VideoProcessor exactly like app/routers/video.py (same options, same model loaders) and records per-stage and
per-call latency by wrapping the hot calls from the outside, so no app code is modified.

    python bench_app_flow.py --video ../demo_images/sample_reef_transect_25fps.mp4 --runs 3
    python bench_app_flow.py --trim-sec 1 --runs 1          # local smoke test only
"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:  # torch is optional here, only used for GPU memory
    torch = None

import core.video.engine as engine  # noqa: E402
import core.video.flow_tracking as flow_tracking  # noqa: E402
from core.models import load_bioclip_model, load_bleaching_model, load_coralscop_model  # noqa: E402

# Options the app sends (VideoProcessRequest defaults)
APP_PARAMS: Dict[str, Any] = {
    "overlap_threshold": 0.70, "min_stride": 10, "max_stride": 25, "points_per_side": 16,
    "pred_iou_thresh": 0.50, "stability_thresh": 0.50, "classify": True, "export_mp4": True,
}

_SAMPLES: Dict[str, List[float]] = defaultdict(list)


def _timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _SAMPLES[name].append(time.perf_counter() - t)
    return wrapper


def install_probes():
    """engine and flow_tracking import these names directly, so patch their namespaces."""
    flow_tracking.run_segmentation = _timed("keyframe_segmentation", flow_tracking.run_segmentation)
    flow_tracking.classify_coral_taxonomy = _timed("bioclip_batch", flow_tracking.classify_coral_taxonomy)
    flow_tracking.detect_coral_bleaching = _timed("yolo_bleaching_batch", flow_tracking.detect_coral_bleaching)
    flow_tracking.FlowTracker.step = _timed("tracker_step (per frame)", flow_tracking.FlowTracker.step)
    engine.labels_to_geometry = _timed("labels_to_geometry", engine.labels_to_geometry)
    engine.run_deliverable_exports = _timed("export_all", engine.run_deliverable_exports)


def summarize_samples() -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in _SAMPLES.items():
        s = sorted(v)
        out[k] = {
            "calls": len(v), "total_sec": round(sum(v), 4), "mean_ms": round(statistics.mean(v) * 1000, 2),
            "p50_ms": round(s[len(s) // 2] * 1000, 2), "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))] * 1000, 2),
            "max_ms": round(s[-1] * 1000, 2),
        }
    return out


def peak_rss_mb() -> float:
    u = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(u / (1024 * 1024) if platform.system() == "Darwin" else u / 1024, 1)


def gpu_info() -> Dict[str, Any]:
    if torch is not None and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"name": p.name, "vram_gib": round(p.total_memory / 2**30, 1)}
    return {"name": "CPU / MPS"}


def run_job(processor, video: str, params: Dict[str, Any]) -> Dict[str, Any]:
    _SAMPLES.clear()
    t0 = time.perf_counter()
    job_id = processor.start_job(video_path=video, params=dict(params))
    while True:
        time.sleep(0.25)
        job = processor.get_job(job_id)
        if job and job.get("state") in (engine.VideoJobState.COMPLETED, engine.VideoJobState.ERROR, engine.VideoJobState.CANCELLED):
            break
    wall = time.perf_counter() - t0
    if job["state"] != engine.VideoJobState.COMPLETED:
        raise RuntimeError(f"job ended in {job['state']}: {job.get('error')}")
    frames, dur = job["total_frames"], job["metadata"]["duration_sec"]
    return {
        "frames": frames, "video_sec": dur, "wall_sec": round(wall, 3), "fps": round(frames / wall, 2),
        "realtime_factor": round(dur / wall, 3), "keyframes": job.get("keyframes_count"),
        "unique_corals": job.get("total_unique_corals"), "bleaching_pct": job.get("bleaching_prevalence_pct"),
        "export_status": job.get("export_status"),
        "exports": {k: bool(v) and os.path.isfile(v) for k, v in (job.get("export_paths") or {}).items()},
        "export_errors": job.get("export_errors"), "stage_timings": job.get("timings") or {},
        "call_stats": summarize_samples(), "peak_rss_mb": peak_rss_mb(),
    }


def markdown(report: Dict[str, Any]) -> str:
    w, t = report["warm_median"], report["warm_median"]["stage_timings"]
    lines = [
        f"### {report['gpu']['name']}  ({report['video']}, {w['frames']} frames, {w['video_sec']}s)",
        "",
        f"- ORT provider: `{report['ort_provider']}`",
        f"- cold run {report['cold']['wall_sec']}s | warm median {w['wall_sec']}s",
        f"- **{w['fps']} FPS** | {w['realtime_factor']}x real time | {w['unique_corals']} colonies | "
        f"{w['keyframes']} keyframes | exports {w['export_status']}",
        "", "| stage | sec |", "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in t.items() if k.endswith("_sec")]
    lines += ["", "| call | n | mean ms | p50 | p95 | max |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {k} | {s['calls']} | {s['mean_ms']} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} |"
              for k, s in sorted(w["call_stats"].items())]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(os.path.dirname(__file__), "..", "demo_images", "sample_reef_transect_25fps.mp4"))
    ap.add_argument("--runs", type=int, default=3, help="1 cold + (runs-1) warm")
    ap.add_argument("--trim-sec", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default="bench_result.json")
    ap.add_argument("--out-md", default="bench_result.md")
    ap.add_argument("--out-video", default=None, help="accepted for the Modal runner; the annotated MP4 is the job export")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if args.trim_sec:
        trimmed = os.path.join(tempfile.mkdtemp(prefix="bench_trim_"), "trim_25fps.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-t", str(args.trim_sec), "-r", "25", "-an", trimmed], check=True)
        video = trimmed

    providers: Dict[str, Any] = {}

    def sam_loader():
        model = load_coralscop_model(device=args.device)
        providers["ort"] = model["encoder"].get_providers()[0]
        return model

    install_probes()
    processor = engine.VideoProcessor(
        sam_loader=sam_loader,
        bioclip_loader=lambda: load_bioclip_model(device=args.device),
        bleaching_loader=lambda: load_bleaching_model(device=args.device),
        cache_dir=tempfile.mkdtemp(prefix="bench_cache_"),
    )
    params = dict(APP_PARAMS, device_pref=args.device)

    runs = []
    for i in range(args.runs):
        r = run_job(processor, video, params)
        runs.append(r)
        print(f"run {i + 1}/{args.runs} ({'cold' if i == 0 else 'warm'}): {r['wall_sec']}s  {r['fps']} FPS", flush=True)

    warm = runs[1:] or runs
    report = {
        "video": os.path.basename(video), "gpu": gpu_info(), "ort_provider": providers.get("ort"), "params": params,
        "cold": runs[0], "warm_median": sorted(warm, key=lambda r: r["wall_sec"])[len(warm) // 2], "all_runs": runs,
    }
    processor.close()
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    md = markdown(report)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
