"""
Standalone video pipeline that propagates coral masks with dense optical flow instead of per-colony SAM2 tracking.

Corals are static and the camera moves, so all colonies live in ONE label image that is warped frame to frame with a
dense flow field (cost does not grow with the colony count). CoralSCOP only runs on keyframes, in a background thread,
and re-anchors the labels when its result arrives (the result is carried to the current frame through the flows
recorded since the keyframe). New colonies are classified once (BioCLIP genus + YOLO bleaching) in another thread.

This is an offline script; it is not wired into the app's video engine.

    python flow_pipeline.py --video ../demo_images/sample_reef_transect_25fps.mp4 --out-video out.mp4 --classify
"""

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import (  # noqa: E402
    HF_SAM_DECODER_FILE,
    HF_SAM_ENCODER_FILE,
    find_or_download_onnx_model,
    load_bioclip_model,
    load_bleaching_model,
    load_coralscop_model,
)
from core.segmentation import run_segmentation  # noqa: E402
from core.taxonomy import classify_coral_taxonomy, detect_coral_bleaching  # noqa: E402
from core.video.keyframe_selector import EgoMotionKeyframeSelector  # noqa: E402

PROXY_H = 480            # working resolution for flow, labels and segmentation
FLOW_SCALE = 0.5         # flow is computed at this fraction of the proxy and upsampled
DECODE_SCALE = 0.5       # CoralSCOP masks are decoded at this fraction of the proxy
IOU_MATCH = 0.30         # keyframe detection <-> colony association threshold (same as the app)
CONTAIN_MATCH = 0.60     # a detection mostly inside a colony (or vice versa) also matches
MAX_MISSED_KEYFRAMES = 4  # a colony is dropped after this many keyframes without a re-detection
MAX_INFLIGHT = 2         # keyframe jobs in flight (one running, one queued); further keyframes are skipped
MAX_LAG_FRAMES = 90      # block on the oldest keyframe job once it is this many frames old
SEG = dict(points_per_side=16, pred_iou_thresh=0.50, stability_score_thresh=0.50, min_mask_region_area=25)
SELECTOR = dict(overlap_threshold=0.70, min_stride=10, max_stride=25)


# --------------------------------------------------------------------------- models
def load_models(device: str, classify: bool) -> Dict[str, Any]:
    """CoralSCOP (fp16 on CUDA, fp32 elsewhere) and, optionally, the classifiers."""
    sam = load_coralscop_model(device=device)
    providers = sam["encoder"].get_providers()
    if "CUDAExecutionProvider" in providers:
        try:
            from core.device import get_cuda_provider_options
            from core.fp16_models import build_fp16_session

            cuda = [("CUDAExecutionProvider", get_cuda_provider_options())]
            sam = {
                "encoder": build_fp16_session(find_or_download_onnx_model(HF_SAM_ENCODER_FILE), cuda),
                "decoder": build_fp16_session(find_or_download_onnx_model(HF_SAM_DECODER_FILE), cuda),
            }
        except Exception as exc:  # noqa: BLE001 - fp16 is an optimisation, fall back to the fp32 sessions
            print(f"fp16 CoralSCOP unavailable, using fp32: {exc}", flush=True)
    models = {"sam": sam, "providers": providers}
    if classify:
        models["bioclip"] = load_bioclip_model(device=device)
        models["bleaching"] = load_bleaching_model(device=device)
    return models


# --------------------------------------------------------------------------- label bookkeeping
def reanchor(
    labels: np.ndarray, detections: List[np.ndarray], next_id: int, missed: Dict[int, int], ious: List[float]
) -> Tuple[np.ndarray, int, int]:
    """Associate fresh keyframe masks with the warped colonies (one-to-one, best score first) and repaint the labels.

    A detection mostly inside a colony (or the reverse) counts as a match because CoralSCOP often returns a sub-part
    or a slightly larger blob of the same colony. Missed colonies stay (and keep warping) for a few keyframes so a
    later re-detection reuses their id. Returns (labels, next_id, number_of_new_colonies).
    """
    areas = np.bincount(labels.ravel(), minlength=next_id + 1)
    d_areas = [int(m.sum()) for m in detections]
    pairs = []  # (score, iou, detection index, colony id)
    for i, m in enumerate(detections):
        counts = np.bincount(labels[m], minlength=areas.size)
        counts[0] = 0
        for cid in np.nonzero(counts)[0]:
            inter = int(counts[cid])
            iou = inter / float(d_areas[i] + int(areas[cid]) - inter)
            contain = inter / float(max(1, min(d_areas[i], int(areas[cid]))))
            score = max(iou, 0.8 * contain if contain >= CONTAIN_MATCH else 0.0)
            if score >= IOU_MATCH:
                pairs.append((score, iou, i, int(cid)))
    pairs.sort(reverse=True)

    det_to_id: Dict[int, int] = {}
    used = set()
    for _, iou, i, cid in pairs:
        if i not in det_to_id and cid not in used:
            det_to_id[i] = cid
            used.add(cid)
            ious.append(iou)
    new = 0
    for i in range(len(detections)):
        if i not in det_to_id:
            det_to_id[i] = next_id
            next_id += 1
            new += 1

    drop = set()
    for cid in set(int(c) for c in np.unique(labels)) - {0}:
        missed[cid] = 0 if cid in used else missed.get(cid, 0) + 1
        if missed[cid] >= MAX_MISSED_KEYFRAMES:
            drop.add(cid)
    lut = np.arange(max(next_id, areas.size) + 1, dtype=np.uint16)  # clears repainted and dropped colonies
    for cid in used | drop:
        lut[cid] = 0
    labels = lut[labels]
    for i in sorted(range(len(detections)), key=lambda k: -d_areas[k]):  # big first, small on top
        labels[detections[i]] = det_to_id[i]
    return labels, next_id, new


def guided_prompts(labels: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Decoder prompts: one point per known colony (fresh mask corrects drift) plus grid points only where nothing is known."""
    free = cv2.dilate((labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    pts = []
    for cid in np.unique(labels):
        if cid == 0:
            continue
        ys, xs = np.nonzero(labels == cid)
        if ys.size < 40:
            continue
        y0, x0 = ys.min(), xs.min()
        crop = (labels[y0 : ys.max() + 1, x0 : xs.max() + 1] == cid).astype(np.uint8)
        py, px = np.unravel_index(int(cv2.distanceTransform(crop, cv2.DIST_L2, 3).argmax()), crop.shape)
        pts.append((x0 + px, y0 + py))
        free[y0 + py, x0 + px] = True
    return (np.asarray(pts, dtype=np.float32), free) if pts else (None, None)


def warp(img: np.ndarray, flow: np.ndarray, grid: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(img, grid[0] + flow[..., 0], grid[1] + flow[..., 1], cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# --------------------------------------------------------------------------- output video
def palette(n: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(7)
    hsv = np.stack([rng.integers(0, 180, n), rng.integers(170, 256, n), rng.integers(200, 256, n)], axis=-1)
    return cv2.cvtColor(hsv.astype(np.uint8).reshape(n, 1, 3), cv2.COLOR_HSV2BGR).reshape(n, 3)


def render_frame(frame, labels, colors, text, is_kf, info):
    """Colour fill + outline built at proxy size, upscaled once; genus / bleaching tags where known."""
    H, W = frame.shape[:2]
    ph, pw = labels.shape
    fg = cv2.resize((labels > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    fill = cv2.resize(colors[labels % len(colors)], (W, H), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    cv2.copyTo(cv2.addWeighted(frame, 0.55, fill, 0.45, 0), fg, out)
    edge = cv2.morphologyEx(labels, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[cv2.resize(edge.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0] = (255, 255, 255)

    if info:
        flat = labels.ravel()
        cnt = np.bincount(flat, minlength=int(flat.max()) + 1)
        yy, xx = np.mgrid[0:ph, 0:pw]
        cx = np.bincount(flat, weights=xx.ravel(), minlength=cnt.size) / np.maximum(cnt, 1)
        cy = np.bincount(flat, weights=yy.ravel(), minlength=cnt.size) / np.maximum(cnt, 1)
        for cid in np.nonzero(cnt)[0]:
            if cid == 0 or cnt[cid] < 400 or cid not in info:
                continue
            tx, ty = int(cx[cid] * W / pw) - 60, int(cy[cid] * H / ph)
            bleached = info[cid]["is_bleached"]
            label = f"#{cid} {info[cid]['genus']}"
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255) if bleached else (255, 255, 255), 2)
            if bleached:
                cv2.putText(out, "BLEACHED", (tx, ty + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)
    cv2.rectangle(out, (0, 0), (560, 46), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if is_kf else (255, 255, 255), 2)
    return out


def open_ffmpeg(path: str, w: int, h: int, fps: float) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to write the output video")
    return subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", path],
        stdin=subprocess.PIPE,
    )


# --------------------------------------------------------------------------- pipeline
def run(video: str, out_video: Optional[str], models: Dict[str, Any]) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    ph, pw = PROXY_H, int(round(W * PROXY_H / H))
    fw, fh = int(pw * FLOW_SCALE), int(ph * FLOW_SCALE)
    classify = "bioclip" in models

    selector = EgoMotionKeyframeSelector(**SELECTOR)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    grid = tuple(np.meshgrid(np.arange(pw, dtype=np.float32), np.arange(ph, dtype=np.float32)))
    colors = palette()

    tot: Dict[str, float] = defaultdict(float)  # seconds per stage (worker threads add too)
    cnt: Dict[str, int] = defaultdict(int)

    def timed_add(name: str, sec: float):
        tot[name] += sec
        cnt[name] += 1

    # ---- output video: rendering + encoding happen in their own thread
    proc = open_ffmpeg(out_video, W, H, fps) if out_video else None
    render_q: "queue.Queue" = queue.Queue(maxsize=8)

    def render_worker():
        while (item := render_q.get()) is not None:
            frame, lab, text, is_kf, info = item
            t0 = time.perf_counter()
            out = render_frame(frame, lab, colors, text, is_kf, info)
            t1 = time.perf_counter()
            proc.stdin.write(out.tobytes())
            timed_add("render_overlay (thread)", t1 - t0)
            timed_add("encode_write (thread)", time.perf_counter() - t1)

    render_thread = threading.Thread(target=render_worker, daemon=True) if proc else None
    if render_thread:
        render_thread.start()

    # ---- keyframe segmentation (thread) and classification (thread)
    seg_pool, cls_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    colony_info: Dict[int, Dict[str, Any]] = {}
    cls_futs: List[Any] = []

    def segment(rgb, pts, mask) -> List[np.ndarray]:
        t0 = time.perf_counter()
        found, _ = run_segmentation(models["sam"], rgb, custom_prompt_points=pts, discovery_mask=mask,
                                    decode_scale=DECODE_SCALE, **SEG)
        dets = [cv2.resize(np.asarray(m["mask"], np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0 for m in found]
        timed_add("keyframe_segmentation (thread)", time.perf_counter() - t0)
        return dets

    def classify_worker(crops, ids, frame_no):
        t0 = time.perf_counter()
        tax = classify_coral_taxonomy(crops, bioclip_bundle=models["bioclip"], top_k=3)
        t1 = time.perf_counter()
        ble = detect_coral_bleaching(crops, bleaching_model=models["bleaching"])
        timed_add("classify_bioclip (thread)", t1 - t0)
        timed_add("classify_yolo_bleaching (thread)", time.perf_counter() - t1)
        for cid, t, b in zip(ids, tax, ble):
            colony_info[cid] = {
                "genus": str(t.get("genus") or "Coral"), "growth_form": str(t.get("growth_form") or "-"),
                "taxon_conf": round(float(t.get("confidence_pct", 0.0)), 1),
                "condition": b.get("status", "Healthy"), "is_bleached": bool(b.get("is_bleached", False)),
                "condition_conf": round(float(b.get("confidence_pct", 0.0)), 1), "first_frame": frame_no,
            }

    def classify_new(first_id, last_id, labels, frame_bgr, frame_no):
        """Classify the colonies created by the latest re-anchor, once, from padded full-res crops."""
        crops, ids = [], []
        for cid in range(first_id, last_id):
            ys, xs = np.nonzero(labels == cid)
            if ys.size < 15:
                continue
            bw, bh = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
            x0, x1 = max(0, xs.min() - 0.1 * bw), min(pw, xs.max() + 1 + 0.1 * bw)
            y0, y1 = max(0, ys.min() - 0.1 * bh), min(ph, ys.max() + 1 + 0.1 * bh)
            X0, X1, Y0, Y1 = int(x0 * W / pw), int(x1 * W / pw), int(y0 * H / ph), int(y1 * H / ph)
            if X1 - X0 >= 10 and Y1 - Y0 >= 10:
                crops.append(Image.fromarray(np.ascontiguousarray(frame_bgr[Y0:Y1, X0:X1, ::-1])))
                ids.append(cid)
        if crops:
            cls_futs.append(cls_pool.submit(classify_worker, crops, ids, frame_no))

    def apply_keyframe(job, labels, next_id, n, frame_bgr):
        """Carry a finished keyframe's detections to frame n through the flows since it was submitted, then re-anchor."""
        dets = job["fut"].result()
        t0 = time.perf_counter()
        det_labels = np.zeros((ph, pw), np.uint16)
        for i in sorted(range(len(dets)), key=lambda k: -int(dets[k].sum())):
            det_labels[dets[i]] = i + 1
        for flow in job["flows"]:
            det_labels = warp(det_labels, flow, grid)
        moved = [m for m in (det_labels == i + 1 for i in range(len(dets))) if m.sum() >= 15]
        first = next_id
        labels, next_id, new = reanchor(labels, moved, next_id, missed, ious)
        lags.append(n - job["start"])
        timed_add("keyframe_reanchor", time.perf_counter() - t0)
        if classify:
            classify_new(first, next_id, labels, frame_bgr, n)
        return labels, next_id

    # ---- main loop
    labels = np.zeros((ph, pw), np.uint16)
    next_id, n, skipped = 1, 0, 0
    missed: Dict[int, int] = {}
    ious: List[float] = []
    lags: List[int] = []
    jobs: List[Dict[str, Any]] = []
    prev_small = frame_bgr = None
    t_wall = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        frame_bgr = frame
        proxy = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(proxy, cv2.COLOR_BGR2RGB)
        small = cv2.resize(cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY), (fw, fh), interpolation=cv2.INTER_AREA)
        timed_add("decode+proxy", time.perf_counter() - t0)

        t0 = time.perf_counter()
        is_kf, _ = selector.evaluate_frame(rgb)
        timed_add("ego_motion_keyframe_select", time.perf_counter() - t0)

        flow = None
        if prev_small is not None:
            t0 = time.perf_counter()
            flow = cv2.resize(dis.calc(small, prev_small, None), (pw, ph), interpolation=cv2.INTER_LINEAR) / FLOW_SCALE
            timed_add("dense_flow", time.perf_counter() - t0)
            t0 = time.perf_counter()
            labels = warp(labels, flow, grid)
            timed_add("warp_labels", time.perf_counter() - t0)
        prev_small = small

        applied = False
        for job in jobs:
            if flow is not None:
                job["flows"].append(flow)
        while jobs and (jobs[0]["fut"].done() or n - jobs[0]["start"] > MAX_LAG_FRAMES):
            t0 = time.perf_counter()
            jobs[0]["fut"].result()
            timed_add("blocked_waiting_for_keyframe", time.perf_counter() - t0)
            labels, next_id = apply_keyframe(jobs.pop(0), labels, next_id, n, frame_bgr)
            applied = True

        if n == 0:  # the first frame has to be segmented synchronously
            first = next_id
            labels, next_id, _ = reanchor(labels, segment(rgb, None, None), next_id, missed, ious)
            lags.append(0)
            applied = True
            if classify:
                classify_new(first, next_id, labels, frame_bgr, n)
        elif is_kf:
            if len(jobs) < MAX_INFLIGHT:
                pts, mask = guided_prompts(labels)
                jobs.append({"fut": seg_pool.submit(segment, rgb, pts, mask), "flows": [], "start": n})
            else:
                skipped += 1

        if proc:
            text = f"frame {n}{'  KEYFRAME' if applied else ''}  colonies {int((np.unique(labels) > 0).sum())}"
            render_q.put((frame_bgr, labels.copy(), text, applied, dict(colony_info)))
        n += 1

    for job in jobs:  # keyframes still running when the video ends
        t0 = time.perf_counter()
        job["fut"].result()
        timed_add("blocked_waiting_for_keyframe", time.perf_counter() - t0)
        labels, next_id = apply_keyframe(job, labels, next_id, n - 1, frame_bgr)
    seg_pool.shutdown(wait=True)
    t0 = time.perf_counter()
    for fut in cls_futs:
        fut.result()
    cls_pool.shutdown(wait=True)
    if classify:
        timed_add("classify_drain (wait at end)", time.perf_counter() - t0)
    if render_thread:
        render_q.put(None)
        render_thread.join()
        proc.stdin.close()
        proc.wait()
    cap.release()
    wall = time.perf_counter() - t_wall

    report = {
        "frames": n, "video_sec": round(n / fps, 2), "resolution": f"{W}x{H}",
        "wall_sec": round(wall, 3), "fps": round(n / wall, 2), "realtime_factor": round(n / fps / wall, 3),
        "stage_sec": {k: round(v, 3) for k, v in tot.items()},
        "stage_ms_per_call": {k: round(1000 * v / cnt[k], 2) for k, v in tot.items()},
        "keyframes_applied": len(lags), "keyframes_skipped": skipped, "keyframe_lag_frames": lags,
        "colonies_seen": next_id - 1,
        "mean_iou_warped_vs_fresh_detection": round(float(np.mean(ious)), 3) if ious else None,
    }
    if classify:
        report["classification"] = {
            "classified": len(colony_info),
            "genus_counts": dict(sorted(Counter(c["genus"] for c in colony_info.values()).items())),
            "bleached": sum(c["is_bleached"] for c in colony_info.values()),
            "colonies": {str(k): v for k, v in sorted(colony_info.items())},
        }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=os.path.join(os.path.dirname(__file__), "..", "demo_images", "sample_reef_transect_25fps.mp4"))
    ap.add_argument("--out-video", default="flow_masked.mp4")
    ap.add_argument("--out-json", default="flow_result.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--classify", action="store_true", help="BioCLIP genus + YOLO bleaching for every new colony")
    ap.add_argument("--runs", type=int, default=2, help="the first run warms up CUDA / ONNX Runtime; the last is reported")
    ap.add_argument("--trim-sec", type=float, default=None, help="only process the first N seconds (smoke tests)")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if args.trim_sec:
        trimmed = os.path.join(tempfile.mkdtemp(), "trim.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-t", str(args.trim_sec), "-r", "25", "-an", trimmed], check=True)
        video = trimmed

    t0 = time.perf_counter()
    models = load_models(args.device, args.classify)
    load_sec = round(time.perf_counter() - t0, 2)

    report = None
    for i in range(args.runs):
        report = run(video, args.out_video, models)
        print(f"run {i + 1}/{args.runs}: {report['wall_sec']}s  {report['fps']} FPS", flush=True)
    report.update(model_load_sec=load_sec, ort_providers=models["providers"])
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("classification", "stage_ms_per_call")}, indent=2))


if __name__ == "__main__":
    main()
