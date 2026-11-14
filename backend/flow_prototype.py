"""
Prototype: camera-motion-driven mask propagation with dense optical flow (no SAM2 tracker).

Corals are static, the camera moves. So instead of tracking every colony with SAM2 (cost grows with colony
count), colonies live in ONE label image that is warped frame to frame with a dense per-pixel flow field.
CoralSCOP only runs on keyframes (same ego-motion schedule as the app) and re-anchors the labels there.

Per-frame cost is independent of the number of colonies.

    python flow_prototype.py --video ../demo_images/sample_reef_transect_25fps.mp4 --out-video out.mp4

Not included (kept out of the timings on purpose): BioCLIP / YOLO classification (async in the app, ~1.6 s on T4).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import load_coralscop_model  # noqa: E402
from core.segmentation import run_segmentation  # noqa: E402
from core.video.keyframe_selector import EgoMotionKeyframeSelector  # noqa: E402

IOU_MATCH = 0.30  # same threshold the app uses for keyframe association
MAX_MISSED_KEYFRAMES = 2  # drop a colony after it was not re-detected on this many consecutive keyframes


def palette(n: int) -> np.ndarray:
    """BGR colour per id (id 0 unused)."""
    rng = np.random.default_rng(7)
    hsv = np.zeros((n, 1, 3), np.uint8)
    hsv[:, 0, 0] = rng.integers(0, 180, n)
    hsv[:, 0, 1] = rng.integers(170, 256, n)
    hsv[:, 0, 2] = rng.integers(200, 256, n)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)


class Timer:
    def __init__(self):
        self.t: Dict[str, float] = defaultdict(float)
        self.n: Dict[str, int] = defaultdict(int)
        self.samples: Dict[str, List[float]] = defaultdict(list)

    def add(self, name: str, sec: float):
        self.t[name] += sec
        self.n[name] += 1
        self.samples[name].append(sec)


class FfmpegWriter:
    def __init__(self, path: str, w: int, h: int, fps: float):
        ff = shutil.which("ffmpeg")
        self.proc = None
        self.cv = None
        if ff:
            cmd = [
                ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def reanchor(
    labels: np.ndarray,
    detections: List[np.ndarray],
    next_id: int,
    missed: Dict[int, int],
    frame_idx: int,
    last_anchor: Dict[int, int],
    drift: List[Tuple[int, float]],
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """Associate fresh keyframe masks with the warped labels (IoU), repaint the label image from detections."""
    areas = np.bincount(labels.ravel(), minlength=next_id + 1)
    matched_ids = set()
    assign: List[Tuple[np.ndarray, int]] = []
    new_count = 0
    # largest first so small colonies are painted on top of big ones
    order = sorted(range(len(detections)), key=lambda i: -int(detections[i].sum()))
    for i in order:
        m = detections[i]
        d_area = int(m.sum())
        ids = np.bincount(labels[m], minlength=areas.size)
        ids[0] = 0
        best = int(ids.argmax()) if ids.size else 0
        iou = 0.0
        if best and best not in matched_ids:
            inter = int(ids[best])
            iou = inter / float(d_area + int(areas[best]) - inter)
        if best and best not in matched_ids and iou >= IOU_MATCH:
            cid = best
            matched_ids.add(cid)
            drift.append((frame_idx - last_anchor.get(cid, frame_idx), iou))
            last_anchor[cid] = frame_idx
        else:
            cid = next_id
            next_id += 1
            new_count += 1
            last_anchor[cid] = frame_idx
            matched_ids.add(cid)
        assign.append((m, cid))

    # age out colonies that were not re-detected
    alive = set(np.unique(labels)) - {0}
    dropped = 0
    for cid in alive:
        if cid in matched_ids:
            missed[cid] = 0
        else:
            missed[cid] = missed.get(cid, 0) + 1
            if missed[cid] >= MAX_MISSED_KEYFRAMES:
                labels[labels == cid] = 0
                dropped += 1
    for m, cid in assign:
        labels[m] = cid
    return labels, next_id, {"new": new_count, "matched": len(matched_ids) - new_count, "dropped": dropped}


def run_once(
    video: str,
    model: Dict[str, Any],
    out_video: Optional[str],
    proxy_h: int,
    opts: Dict[str, Any],
) -> Dict[str, Any]:
    tm = Timer()
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    pw = int(round(W * proxy_h / H))
    ph = proxy_h

    selector = EgoMotionKeyframeSelector(
        overlap_threshold=opts["overlap_threshold"], min_stride=opts["min_stride"], max_stride=opts["max_stride"]
    )
    dis = cv2.DISOpticalFlow_create(
        {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
         "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[opts["flow_preset"]]
    )
    gx, gy = np.meshgrid(np.arange(pw, dtype=np.float32), np.arange(ph, dtype=np.float32))

    writer = FfmpegWriter(out_video, W, H, fps) if out_video else None
    colors = palette(4096)

    labels = np.zeros((ph, pw), np.uint16)
    prev_gray = None
    next_id = 1
    missed: Dict[int, int] = {}
    last_anchor: Dict[int, int] = {}
    drift: List[Tuple[int, float]] = []
    kf_info: List[Dict[str, Any]] = []
    counts: List[int] = []
    n = 0
    t_wall0 = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            break
        proxy_bgr = cv2.resize(frame_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        proxy_rgb = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2GRAY)
        tm.add("decode+proxy", time.perf_counter() - t0)

        t0 = time.perf_counter()
        is_kf, meta = selector.evaluate_frame(proxy_rgb)
        tm.add("ego_motion_keyframe_select", time.perf_counter() - t0)

        if prev_gray is not None:
            t0 = time.perf_counter()
            flow = dis.calc(gray, prev_gray, None)  # for each current pixel: where it was in the previous frame
            tm.add("dense_flow(DIS)", time.perf_counter() - t0)
            t0 = time.perf_counter()
            labels = cv2.remap(
                labels, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            tm.add("warp_labels(remap)", time.perf_counter() - t0)

        info = None
        if n == 0 or is_kf:
            t0 = time.perf_counter()
            masks_info, _ = run_segmentation(
                model=model, image=proxy_rgb, points_per_side=opts["points_per_side"],
                pred_iou_thresh=0.50, stability_score_thresh=0.50, min_mask_region_area=25,
            )
            tm.add("keyframe_segmentation(CoralSCOP)", time.perf_counter() - t0)
            t0 = time.perf_counter()
            dets = [np.asarray(m["mask"], dtype=bool) for m in masks_info]
            labels, next_id, info = reanchor(labels, dets, next_id, missed, n, last_anchor, drift)
            tm.add("keyframe_match_reanchor", time.perf_counter() - t0)
            info.update(frame=n, detections=len(dets))
            kf_info.append(info)
        prev_gray = gray

        present = np.unique(labels)
        counts.append(int((present > 0).sum()))

        if writer is not None:
            t0 = time.perf_counter()
            lab_full = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST)
            fg = lab_full > 0
            out = frame_bgr.copy()
            col = colors[lab_full % 4096]
            out[fg] = (0.55 * frame_bgr[fg] + 0.45 * col[fg]).astype(np.uint8)
            edge = cv2.morphologyEx(labels, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            edge_full = cv2.resize(edge.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            out[edge_full] = (255, 255, 255)
            cv2.rectangle(out, (0, 0), (560, 46), (0, 0, 0), -1)
            cv2.putText(
                out, f"frame {n}{'  KEYFRAME' if info else ''}  colonies {counts[-1]}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if info else (255, 255, 255), 2,
            )
            tm.add("render_overlay", time.perf_counter() - t0)
            t0 = time.perf_counter()
            writer.write(out)
            tm.add("encode_write(ffmpeg)", time.perf_counter() - t0)
        n += 1

    t0 = time.perf_counter()
    if writer is not None:
        writer.close()
        tm.add("encode_flush", time.perf_counter() - t0)
    cap.release()
    wall = time.perf_counter() - t_wall0

    age_bins = {"1-10": [], "11-20": [], "21-30": [], ">30": []}
    for age, iou in drift:
        key = "1-10" if age <= 10 else "11-20" if age <= 20 else "21-30" if age <= 30 else ">30"
        age_bins[key].append(iou)
    return {
        "frames": n,
        "video_sec": round(n / fps, 2),
        "resolution": f"{W}x{H}",
        "flow_resolution": f"{pw}x{ph}",
        "wall_sec": round(wall, 3),
        "e2e_fps": round(n / wall, 2),
        "realtime_factor": round((n / fps) / wall, 3),
        "stage_sec": {k: round(v, 3) for k, v in tm.t.items()},
        "stage_ms_per_call": {k: round(1000 * v / tm.n[k], 2) for k, v in tm.t.items()},
        "stage_calls": dict(tm.n),
        "stage_pct_wall": {k: round(100 * v / wall, 1) for k, v in tm.t.items()},
        "keyframes": kf_info,
        "colonies_seen": next_id - 1,
        "colonies_per_frame_max": max(counts) if counts else 0,
        "reanchor_iou_mean": round(float(np.mean([d[1] for d in drift])), 3) if drift else None,
        "reanchor_iou_by_age_frames": {
            k: {"n": len(v), "mean_iou": round(float(np.mean(v)), 3) if v else None} for k, v in age_bins.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(os.path.dirname(__file__), "..", "demo_images", "sample_reef_transect_25fps.mp4"))
    ap.add_argument("--out-video", default="flow_masked.mp4")
    ap.add_argument("--out-json", default="flow_result.json")
    ap.add_argument("--passes", type=int, default=2, help="pass 1 is warm-up (CUDA/ORT init); last pass is reported")
    ap.add_argument("--proxy-h", type=int, default=480)
    ap.add_argument("--flow-preset", default="medium", choices=["ultrafast", "fast", "medium"])
    ap.add_argument("--trim-sec", type=float, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if args.trim_sec:
        dst = os.path.join(tempfile.mkdtemp(), "trim.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-t", str(args.trim_sec), "-r", "25", "-an", dst], check=True)
        video = dst

    t0 = time.perf_counter()
    model = load_coralscop_model(device=args.device)
    load_sec = round(time.perf_counter() - t0, 3)
    providers = model["encoder"].get_providers()

    opts = {
        "overlap_threshold": 0.70, "min_stride": 10, "max_stride": 25, "points_per_side": 16,
        "flow_preset": args.flow_preset,
    }
    results = []
    for p in range(args.passes):
        print(f"=== pass {p + 1}/{args.passes} ===", flush=True)
        results.append(run_once(video, model, args.out_video, args.proxy_h, opts))
        print(f"  wall {results[-1]['wall_sec']}s  e2e {results[-1]['e2e_fps']} FPS", flush=True)

    report = {"coralscop_load_sec": load_sec, "ort_providers": providers, "opts": opts, "first_pass": results[0], "final_pass": results[-1]}
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    r = results[-1]
    print(json.dumps({k: r[k] for k in ("frames", "wall_sec", "e2e_fps", "realtime_factor", "stage_sec", "stage_pct_wall", "colonies_seen", "reanchor_iou_mean", "reanchor_iou_by_age_frames")}, indent=2))


if __name__ == "__main__":
    main()
