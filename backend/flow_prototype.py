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
import queue
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import load_coralscop_model  # noqa: E402
from core.segmentation import run_segmentation  # noqa: E402
from core.video.keyframe_selector import EgoMotionKeyframeSelector  # noqa: E402

IOU_MATCH = 0.30  # same threshold the app uses for keyframe association
MAX_ASYNC_LAG = 90  # frames a keyframe result may lag before the main loop blocks on it
MAX_MISSED_KEYFRAMES = 4  # drop a colony after it was not re-detected on this many consecutive keyframes
CONTAIN_MATCH = 0.60  # detection/colony containment that also counts as a match


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
    """Associate fresh keyframe masks with the warped labels, then repaint the label image.

    Matching is one-to-one and globally greedy by score (not per-detection argmax), and a detection that is
    mostly *inside* a warped colony (or vice versa) also counts, because CoralSCOP often returns a sub-part or a
    slightly larger blob of the same colony. Colonies that are missed are kept (warped, not dropped) for several
    keyframes so a re-detection reuses the old id instead of spawning a new one.
    """
    areas = np.bincount(labels.ravel(), minlength=next_id + 1)
    d_areas = [int(m.sum()) for m in detections]
    pairs: List[Tuple[float, float, int, int]] = []  # (score, iou, det_idx, colony_id)
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
    used: set = set()
    for score, iou, i, cid in pairs:
        if i in det_to_id or cid in used:
            continue
        det_to_id[i] = cid
        used.add(cid)
        drift.append((frame_idx - last_anchor.get(cid, frame_idx), iou))
        last_anchor[cid] = frame_idx
    new_count = 0
    for i in range(len(detections)):
        if i not in det_to_id:
            det_to_id[i] = next_id
            last_anchor[next_id] = frame_idx
            next_id += 1
            new_count += 1

    # age bookkeeping + drop colonies missed for too long
    alive = set(int(c) for c in np.unique(labels)) - {0}
    drop = set()
    for cid in alive:
        if cid in used:
            missed[cid] = 0
        else:
            missed[cid] = missed.get(cid, 0) + 1
            if missed[cid] >= MAX_MISSED_KEYFRAMES:
                drop.add(cid)
    # clear old pixels of matched (about to be repainted) and dropped colonies in one LUT pass
    lut = np.arange(max(next_id, areas.size) + 1, dtype=np.uint16)
    for cid in used | drop:
        lut[cid] = 0
    labels = lut[labels]
    for i in sorted(range(len(detections)), key=lambda k: -d_areas[k]):  # big first, small on top
        labels[detections[i]] = det_to_id[i]
    return labels, next_id, {"new": new_count, "matched": len(used), "dropped": len(drop), "kept_stale": len(alive - used - drop)}


def guided_prompts(labels: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Prompt the keyframe decoder with (a) one point per known colony and (b) grid points only where nothing is known.

    Existing colonies still get a fresh mask (so the keyframe corrects flow drift) but the many grid points that
    would just re-segment already-covered reef are skipped, so the decoder runs far fewer batches.
    """
    h, w = labels.shape
    covered = cv2.dilate((labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    mask = ~covered
    pts = []
    for cid in np.unique(labels):
        if cid == 0:
            continue
        ys, xs = np.nonzero(labels == cid)
        if ys.size < 40:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = (labels[y0:y1, x0:x1] == cid).astype(np.uint8)
        dt = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
        py, px = np.unravel_index(int(dt.argmax()), dt.shape)
        pts.append((x0 + px, y0 + py))
        mask[y0 + py, x0 + px] = True
    if not pts:
        return None, None
    return np.asarray(pts, dtype=np.float32), mask


def render_frame(frame_bgr, labels, colors, W, H, text, is_kf):
    """Cheap overlay: colour + outline are built at proxy size and upscaled once; blending is a single addWeighted."""
    fg = (labels > 0).astype(np.uint8)
    col_full = cv2.resize(colors[labels % 4096], (W, H), interpolation=cv2.INTER_NEAREST)
    fg_full = cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(frame_bgr, 0.55, col_full, 0.45, 0)
    out = frame_bgr.copy()
    cv2.copyTo(blended, fg_full, out)
    edge = cv2.morphologyEx(labels, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge_full = cv2.resize((edge > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    out[edge_full > 0] = (255, 255, 255)
    cv2.rectangle(out, (0, 0), (560, 46), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if is_kf else (255, 255, 255), 2)
    return out


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

    def _render_and_write(frame_bgr, lab, text, is_kf):
        t0 = time.perf_counter()
        out = render_frame(frame_bgr, lab, colors, W, H, text, is_kf)
        tm.add("render_overlay", time.perf_counter() - t0)
        t0 = time.perf_counter()
        writer.write(out)
        tm.add("encode_write(ffmpeg)", time.perf_counter() - t0)

    render_q: "queue.Queue" = queue.Queue(maxsize=8)

    def _render_worker():
        while True:
            item = render_q.get()
            if item is None:
                return
            _render_and_write(*item)

    render_thread = None
    if writer is not None and opts["render_thread"]:
        render_thread = threading.Thread(target=_render_worker, daemon=True)
        render_thread.start()

    labels = np.zeros((ph, pw), np.uint16)
    prev_gray = None
    prev_gray_s = None
    fs = float(opts["flow_scale"])
    next_id = 1
    missed: Dict[int, int] = {}
    last_anchor: Dict[int, int] = {}
    drift: List[Tuple[int, float]] = []
    kf_info: List[Dict[str, Any]] = []
    counts: List[int] = []
    seg_pool = ThreadPoolExecutor(max_workers=1)
    pending = None

    def segment(rgb, g_pts, g_mask):
        t0 = time.perf_counter()
        masks_info, _ = run_segmentation(
            model=model, image=rgb, points_per_side=opts["points_per_side"],
            pred_iou_thresh=0.50, stability_score_thresh=0.50, min_mask_region_area=25,
            custom_prompt_points=g_pts, discovery_mask=g_mask,
            decode_scale=opts["decode_scale"], decoder_batch_size=opts["decoder_batch"],
        )
        dets = [np.asarray(m["mask"], dtype=bool) for m in masks_info]
        if opts["decode_scale"] != 1.0:
            dets = [cv2.resize(d.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0 for d in dets]
        tm.add("keyframe_segmentation(CoralSCOP)", time.perf_counter() - t0)
        return dets

    def apply_async(dets, pend, lab, nid, frame_n):
        """Detections were computed on the keyframe at pend['start']; carry them to the current frame through the
        flow fields accumulated since, then re-anchor as usual."""
        t0 = time.perf_counter()
        det_lab = np.zeros((ph, pw), np.uint16)
        for i in sorted(range(len(dets)), key=lambda k: -int(dets[k].sum())):
            det_lab[dets[i]] = i + 1
        for fl in pend["flows"]:
            det_lab = cv2.remap(det_lab, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        moved = [det_lab == i + 1 for i in range(len(dets))]
        moved = [m for m in moved if m.sum() >= 15]
        lab, nid, inf = reanchor(lab, moved, nid, missed, frame_n, last_anchor, drift)
        tm.add("keyframe_match_reanchor(+warp dets)", time.perf_counter() - t0)
        inf.update(frame=frame_n, submitted=pend["start"], detections=len(moved), lag=frame_n - pend["start"])
        kf_info.append(inf)
        return lab, nid, inf

    n = 0
    t_wall0 = time.perf_counter()
    flow = None

    while True:
        t0 = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            break
        proxy_bgr = cv2.resize(frame_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        proxy_rgb = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.resize(gray, (int(pw * fs), int(ph * fs)), interpolation=cv2.INTER_AREA) if fs != 1.0 else gray
        tm.add("decode+proxy", time.perf_counter() - t0)

        t0 = time.perf_counter()
        is_kf, meta = selector.evaluate_frame(proxy_rgb)
        tm.add("ego_motion_keyframe_select", time.perf_counter() - t0)

        flow = None
        if prev_gray is not None:
            t0 = time.perf_counter()
            if fs != 1.0:
                f_small = dis.calc(gray_s, prev_gray_s, None)
                flow = cv2.resize(f_small, (pw, ph), interpolation=cv2.INTER_LINEAR) * (1.0 / fs)
            else:
                flow = dis.calc(gray, prev_gray, None)  # for each current pixel: where it was in the previous frame
            tm.add("dense_flow(DIS)", time.perf_counter() - t0)
            t0 = time.perf_counter()
            labels = cv2.remap(
                labels, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            tm.add("warp_labels(remap)", time.perf_counter() - t0)

        info = None
        if pending is not None:
            if flow is not None:
                pending["flows"].append(flow)
            if pending["fut"].done() or (n - pending["start"]) > MAX_ASYNC_LAG:
                t0 = time.perf_counter()
                dets = pending["fut"].result()
                tm.add("async_wait(blocked)", time.perf_counter() - t0)
                labels, next_id, info = apply_async(dets, pending, labels, next_id, n)
                pending = None
        if n == 0 or (is_kf and not (opts["async_keyframes"] and pending is not None)):
            g_pts, g_mask = guided_prompts(labels) if (opts["guided_prompts"] and n > 0) else (None, None)
            if opts["async_keyframes"] and n > 0:
                pending = {"fut": seg_pool.submit(segment, proxy_rgb.copy(), g_pts, g_mask), "flows": [], "start": n}
            else:
                dets = segment(proxy_rgb, g_pts, g_mask)
                t0 = time.perf_counter()
                labels, next_id, info = reanchor(labels, dets, next_id, missed, n, last_anchor, drift)
                tm.add("keyframe_match_reanchor", time.perf_counter() - t0)
                info.update(frame=n, detections=len(dets), lag=0)
                kf_info.append(info)
        flow = None
        prev_gray = gray
        prev_gray_s = gray_s

        present = np.unique(labels)
        counts.append(int((present > 0).sum()))

        if writer is not None:
            text = f"frame {n}{'  KEYFRAME' if info else ''}  colonies {counts[-1]}"
            if opts["render_thread"]:
                render_q.put((frame_bgr, labels.copy(), text, bool(info)))
            else:
                _render_and_write(frame_bgr, labels, text, bool(info))
        n += 1

    if pending is not None:
        t0 = time.perf_counter()
        dets = pending["fut"].result()
        tm.add("async_wait(blocked)", time.perf_counter() - t0)
        labels, next_id, _ = apply_async(dets, pending, labels, next_id, n - 1)
    seg_pool.shutdown(wait=True)
    t0 = time.perf_counter()
    if render_thread is not None:
        render_q.put(None)
        render_thread.join()
        tm.add("render_drain(wait)", time.perf_counter() - t0)
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
    ap.add_argument("--flow-scale", type=float, default=1.0, help="compute flow at this fraction of the proxy size")
    ap.add_argument("--guided-prompts", action="store_true", help="prompt keyframe SAM decoder at known colonies + uncovered area only")
    ap.add_argument("--no-render-thread", action="store_true")
    ap.add_argument("--async-keyframes", action="store_true", help="run CoralSCOP in a worker thread; apply result later via flow")
    ap.add_argument("--decode-scale", type=float, default=1.0)
    ap.add_argument("--decoder-batch", type=int, default=None)
    ap.add_argument("--fp16-encoder", action="store_true")
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
    if args.fp16_encoder:
        from core.device import get_cuda_provider_options
        from core.fp16_encoder import build_fp16_session
        from core.models import HF_SAM_ENCODER_FILE, find_or_download_onnx_model

        prov = [("CUDAExecutionProvider", get_cuda_provider_options())] if "CUDAExecutionProvider" in providers else providers
        from core.models import HF_SAM_DECODER_FILE
        model = {
            "encoder": build_fp16_session(find_or_download_onnx_model(HF_SAM_ENCODER_FILE), prov),
            "decoder": build_fp16_session(find_or_download_onnx_model(HF_SAM_DECODER_FILE), prov),
        }

    opts = {
        "overlap_threshold": 0.70, "min_stride": 10, "max_stride": 25, "points_per_side": 16,
        "flow_preset": args.flow_preset, "flow_scale": args.flow_scale,
        "guided_prompts": args.guided_prompts, "render_thread": not args.no_render_thread,
        "async_keyframes": args.async_keyframes, "decode_scale": args.decode_scale, "decoder_batch": args.decoder_batch,
        "fp16_encoder": args.fp16_encoder,
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
