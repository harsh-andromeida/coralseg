"""
App-flow benchmark for the ReefVision video pipeline.

Drives VideoProcessor exactly like app/routers/video.py + the UploadView does (same options,
same loaders), and adds per-call latency stats by wrapping the hot functions from the outside,
so no app code is modified.

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
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

import core.video.engine as engine  # noqa: E402
from core.models import load_bioclip_model, load_bleaching_model, load_coralscop_model  # noqa: E402
from core.video.sam2_tracker import SAM2BenthicTracker  # noqa: E402

# Options the UI actually sends (UploadView.handleStartVideoAnalysis) + schema defaults.
APP_PARAMS: Dict[str, Any] = {
    "overlap_threshold": 0.70,
    "min_stride": 10,
    "max_stride": 25,
    "points_per_side": 16,
    "iou_match_threshold": 0.30,
    "pred_iou_thresh": 0.50,
    "stability_thresh": 0.50,
    "tracker_mode": "sam2",
    "model_profile": "sam2",
    "model_size": 512,
    "precision": None,
    "worker_count": 1,
    "compile_tracker": False,
    "export_mp4": True,
}

# ---------------------------------------------------------------- latency recording
_SAMPLES: Dict[str, List[float]] = defaultdict(list)


def _rec(name: str, sec: float):
    _SAMPLES[name].append(sec)


def _timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _rec(name, time.perf_counter() - t)
    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def install_probes():
    """Wrap hot calls once. engine imports these names directly, so patch engine's namespace."""
    engine.run_segmentation = _timed("keyframe_segmentation", engine.run_segmentation)
    engine.classify_coral_taxonomy = _timed("bioclip_taxonomy_batch", engine.classify_coral_taxonomy)
    engine.detect_coral_bleaching = _timed("yolo_bleaching_batch", engine.detect_coral_bleaching)
    engine.crop_track_regions = _timed("crop_regions", engine.crop_track_regions)
    engine.run_deliverable_exports = _timed("export_all", engine.run_deliverable_exports)

    for meth in ("init_video", "register_mask", "retire_objects", "load"):
        setattr(SAM2BenthicTracker, meth, _timed(f"sam2_{meth}", getattr(SAM2BenthicTracker, meth)))

    orig_prop = SAM2BenthicTracker.propagate

    def propagate(self, *a, **k):
        # time spent producing each frame, excluding the consumer's loop body
        t = time.perf_counter()
        for item in orig_prop(self, *a, **k):
            _rec("sam2_propagate_per_frame", time.perf_counter() - t)
            yield item
            t = time.perf_counter()

    SAM2BenthicTracker.propagate = propagate


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def summarize_samples() -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in _SAMPLES.items():
        out[k] = {
            "calls": len(v),
            "total_sec": round(sum(v), 4),
            "mean_ms": round(statistics.mean(v) * 1000, 2),
            "p50_ms": round(pct(v, 50) * 1000, 2),
            "p95_ms": round(pct(v, 95) * 1000, 2),
            "max_ms": round(max(v) * 1000, 2),
        }
    return out


# ---------------------------------------------------------------- environment
def peak_rss_mb() -> float:
    u = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(u / (1024 * 1024) if platform.system() == "Darwin" else u / 1024, 1)


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda": False}
    if torch is not None and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(cuda=True, name=p.name, vram_gib=round(p.total_memory / 2**30, 1), torch=torch.__version__)
    elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info.update(name="Apple MPS", torch=torch.__version__)
    return info


def trim_video(src: str, seconds: float) -> str:
    dst = os.path.join(tempfile.mkdtemp(prefix="bench_trim_"), f"trim_{seconds}s_25fps.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-t", str(seconds), "-r", "25", "-an", dst],
        check=True,
    )
    return dst


# ---------------------------------------------------------------- one job
def run_job(processor, video_path: str, params: Dict[str, Any], load_times: Dict[str, float]) -> Dict[str, Any]:
    _SAMPLES.clear()
    load_times.clear()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    job_id = processor.start_job(video_path=video_path, params=dict(params))
    t_submit = time.perf_counter() - t0

    job = None
    while True:
        time.sleep(0.25)
        job = processor.get_job(job_id)
        if job and job.get("state") in (
            engine.VideoJobState.COMPLETED,
            engine.VideoJobState.ERROR,
            engine.VideoJobState.CANCELLED,
        ):
            break
    wall = time.perf_counter() - t0
    if job["state"] != engine.VideoJobState.COMPLETED:
        raise RuntimeError(f"job ended in {job['state']}: {job.get('error')}")

    timings = job.get("timings") or {}
    frames = job["total_frames"]
    dur = job["metadata"]["duration_sec"]
    pipe = timings.get("total_elapsed_sec", wall)
    mp4 = timings.get("export_annotated_mp4_sec", 0.0)
    export = timings.get("export_deliverables_sec", 0.0) + mp4
    res = {
        "frames": frames,
        "video_sec": dur,
        "wall_sec": round(wall, 3),
        "submit_standardize_sec": round(t_submit, 3),
        "model_load_sec": {k: round(v, 3) for k, v in load_times.items()},
        "pipeline_sec": pipe,
        "e2e_fps": round(frames / wall, 2),
        "pipeline_fps": round(frames / pipe, 2),
        "analysis_fps_excl_export": round(frames / max(pipe - export, 1e-3), 2),
        "tracker_fps": job.get("tracker_fps"),
        "realtime_factor": round(dur / wall, 3),
        "keyframes": job.get("keyframes_count"),
        "unique_corals": job.get("total_unique_corals"),
        "export_status": job.get("export_status"),
        "exports": {k: bool(v) and os.path.isfile(v) for k, v in (job.get("export_paths") or {}).items()},
        "export_errors": job.get("export_errors"),
        "stage_timings": timings,
        "call_stats": summarize_samples(),
        "peak_rss_mb": peak_rss_mb(),
    }
    if torch is not None and torch.cuda.is_available():
        res["peak_vram_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return res


# ---------------------------------------------------------------- report
def markdown(report: Dict[str, Any]) -> str:
    w = report["warm_median"]
    g = report["gpu"]
    L = [
        f"### {g.get('name', 'CPU')}  ({report['video']}, {w['frames']} frames, {w['video_sec']}s)",
        "",
        f"- ORT providers: `{report['ort_providers']}`   tracker device: `{report['tracker_device']}` "
        f"dtype: `{report['tracker_dtype']}`",
        f"- cold run wall {report['cold']['wall_sec']}s  |  warm median wall {w['wall_sec']}s",
        f"- **e2e {w['e2e_fps']} FPS** | pipeline {w['pipeline_fps']} FPS | analysis (no export) "
        f"{w['analysis_fps_excl_export']} FPS | tracker {w['tracker_fps']} FPS | realtime x{w['realtime_factor']}",
        f"- keyframes {w['keyframes']} | corals {w['unique_corals']} | peak VRAM {w.get('peak_vram_alloc_mib', '-')} MiB",
        f"- cold model load: {report['cold']['model_load_sec']}",
        "",
        "| stage | sec | % wall | ms/frame |",
        "|---|---:|---:|---:|",
    ]
    t = w["stage_timings"]
    rows = [
        ("submit + 25fps standardise", w["submit_standardize_sec"]),
        ("decode / ego-motion prep", t.get("video_decode_prep_sec", 0)),
        ("SAM2 propagation (GPU)", t.get("synchronized_gpu_inference_sec", 0)),
        ("mask transfer GPU->CPU", t.get("mask_transfer_sec", 0)),
        ("geometry (contours)", t.get("geometry_extraction_sec", 0)),
        ("keyframe discovery (CoralSCOP)", t.get("keyframe_discovery_sec", 0)),
        ("classification (async BioCLIP+YOLO)", t.get("background_classification_sec", 0)),
        ("persistence (SQLite)", t.get("persistence_sec", 0)),
        ("export CSV+COCO", t.get("export_deliverables_sec", 0)),
        ("export annotated MP4", t.get("export_annotated_mp4_sec", 0)),
        ("unaccounted / idle", t.get("unaccounted_idle_sec", 0)),
    ]
    for name, sec in rows:
        L.append(f"| {name} | {sec:.3f} | {100 * sec / w['wall_sec']:.1f} | {1000 * sec / w['frames']:.2f} |")
    L += ["", "| call | n | mean ms | p50 | p95 | max |", "|---|---:|---:|---:|---:|---:|"]
    for k, s in sorted(w["call_stats"].items()):
        L.append(f"| {k} | {s['calls']} | {s['mean_ms']} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(os.path.dirname(__file__), "..", "demo_images", "sample_reef_transect_25fps.mp4"))
    ap.add_argument("--runs", type=int, default=3, help="1 cold + (runs-1) warm")
    ap.add_argument("--trim-sec", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--override", action="append", default=[], help="k=v param override (json value)")
    ap.add_argument("--out-json", default="bench_result.json")
    ap.add_argument("--out-md", default="bench_result.md")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if args.trim_sec:
        video = trim_video(video, args.trim_sec)

    params = dict(APP_PARAMS, device_pref=args.device)
    for kv in args.override:
        k, v = kv.split("=", 1)
        params[k] = json.loads(v)

    load_times: Dict[str, float] = {}
    providers: Dict[str, Any] = {}

    def wrap_loader(name, fn):
        def loader():
            t = time.perf_counter()
            r = fn(device=args.device)
            load_times[name] = load_times.get(name, 0.0) + time.perf_counter() - t
            if name == "coralscop" and isinstance(r, dict):
                providers["ort"] = r["encoder"].get_providers()
            return r
        return loader

    install_probes()
    cache = tempfile.mkdtemp(prefix="bench_cache_")
    processor = engine.VideoProcessor(
        sam_loader=wrap_loader("coralscop", load_coralscop_model),
        bioclip_loader=wrap_loader("bioclip", load_bioclip_model),
        bleaching_loader=wrap_loader("bleaching", load_bleaching_model),
        cache_dir=cache,
    )

    runs = []
    for i in range(args.runs):
        print(f"=== run {i + 1}/{args.runs} ({'cold' if i == 0 else 'warm'}) ===", flush=True)
        r = run_job(processor, video, params, load_times)
        runs.append(r)
        print(f"  wall {r['wall_sec']}s  e2e {r['e2e_fps']} FPS  tracker {r['tracker_fps']} FPS", flush=True)

    tracker = next(iter(processor._tracker_cache.values()), None)
    warm = runs[1:] or runs
    median_run = sorted(warm, key=lambda r: r["wall_sec"])[len(warm) // 2]
    report = {
        "video": os.path.basename(video),
        "gpu": gpu_info(),
        "ort_providers": providers.get("ort"),
        "tracker_device": getattr(tracker, "device", None),
        "tracker_dtype": str(getattr(tracker, "dtype", None)),
        "params": params,
        "cold": runs[0],
        "warm_median": median_run,
        "all_runs": runs,
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
