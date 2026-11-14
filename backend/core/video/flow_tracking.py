"""
Camera-motion-driven coral tracking for underwater transect video.

Corals are static and the camera moves, so instead of tracking every colony individually, all colonies live in ONE
label image that is warped frame to frame with a dense optical-flow field (cost does not depend on the colony count).
CoralSCOP only runs on ego-motion keyframes, in a background thread, and re-anchors the labels when its result arrives:
the result is carried to the current frame through the flow fields recorded since the keyframe. New colonies are
classified once (BioCLIP genus + YOLO bleaching) in another background thread.

    tracker = FlowTracker(sam_model, width, height, classify={"bioclip": ..., "bleaching": ...})
    for frame_bgr in frames:
        step = tracker.step(frame_bgr)      # step.labels: uint16 colony ids at proxy resolution (0 = background)
        done = tracker.drain_classifications()
    tracker.finish()
"""

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from core.segmentation import run_segmentation
from core.taxonomy import classify_coral_taxonomy, detect_coral_bleaching
from core.video.keyframe_selector import EgoMotionKeyframeSelector

PROXY_H = 480             # working resolution for flow, labels and segmentation
FLOW_SCALE = 0.5          # flow is computed at this fraction of the proxy and upsampled
DECODE_SCALE = 0.5        # CoralSCOP masks are decoded at this fraction of the proxy
IOU_MATCH = 0.30          # keyframe detection <-> colony association threshold
CONTAIN_MATCH = 0.60      # a detection mostly inside a colony (or vice versa) also matches
MAX_MISSED_KEYFRAMES = 4  # a colony is dropped after this many keyframes without a re-detection
MAX_INFLIGHT = 2          # keyframe jobs in flight (one running, one queued); further keyframes are skipped
MAX_LAG_FRAMES = 90       # block on the oldest keyframe job once it is this many frames old


# --------------------------------------------------------------------------- label bookkeeping
def reanchor(
    labels: np.ndarray, detections: List[np.ndarray], next_id: int, missed: Dict[int, int], ious: Optional[List[float]] = None
) -> Tuple[np.ndarray, int]:
    """Associate fresh keyframe masks with the warped colonies (one-to-one, best score first) and repaint the labels.

    A detection mostly inside a colony (or the reverse) counts as a match because CoralSCOP often returns a sub-part
    or a slightly larger blob of the same colony. Missed colonies stay (and keep warping) for a few keyframes so a
    later re-detection reuses their id. Returns (labels, next_id).
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
            if ious is not None:
                ious.append(iou)
    for i in range(len(detections)):
        if i not in det_to_id:
            det_to_id[i] = next_id
            next_id += 1

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
    return labels, next_id


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


# --------------------------------------------------------------------------- tracker
@dataclass
class FlowStep:
    frame_idx: int
    labels: np.ndarray                 # uint16 colony ids at proxy resolution; do not modify
    is_keyframe: bool                  # the ego-motion selector fired on this frame
    keyframe_applied: bool             # a CoralSCOP result was merged into the labels on this frame
    motion_dx: float = 0.0
    motion_dy: float = 0.0
    new_ids: List[int] = field(default_factory=list)


class FlowTracker:
    def __init__(
        self,
        sam_model: Dict[str, Any],
        width: int,
        height: int,
        classify: Optional[Dict[str, Any]] = None,
        overlap_threshold: float = 0.70,
        min_stride: int = 10,
        max_stride: int = 25,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.50,
        stability_thresh: float = 0.50,
    ):
        self.sam = sam_model
        self.classify = classify  # {"bioclip": bundle, "bleaching": session} or None
        self.W, self.H = width, height
        self.ph, self.pw = PROXY_H, int(round(width * PROXY_H / height))
        self.fw, self.fh = int(self.pw * FLOW_SCALE), int(self.ph * FLOW_SCALE)
        self._seg_kwargs = dict(
            points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_thresh, min_mask_region_area=25,
        )
        self.selector = EgoMotionKeyframeSelector(
            overlap_threshold=overlap_threshold, min_stride=min_stride, max_stride=max_stride
        )
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._grid = tuple(np.meshgrid(np.arange(self.pw, dtype=np.float32), np.arange(self.ph, dtype=np.float32)))

        self.labels = np.zeros((self.ph, self.pw), np.uint16)
        self.next_id = 1
        self.n = 0
        self.keyframes_skipped = 0
        self._missed: Dict[int, int] = {}
        self._prev_small: Optional[np.ndarray] = None
        self._jobs: List[Dict[str, Any]] = []
        self._seg_pool = ThreadPoolExecutor(1)
        self._cls_pool = ThreadPoolExecutor(1)
        self._cls_futs: List[Any] = []
        self._cls_done: List[Tuple[int, Dict[str, Any]]] = []
        self._cls_lock = threading.Lock()
        self.timings: Dict[str, float] = defaultdict(float)   # seconds per stage (worker threads add too)

    # ---- helpers
    def _warp(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self._grid[0] + flow[..., 0], self._grid[1] + flow[..., 1], cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _segment(self, rgb: np.ndarray, pts, mask) -> List[np.ndarray]:
        t0 = time.perf_counter()
        found, _ = run_segmentation(self.sam, rgb, custom_prompt_points=pts, discovery_mask=mask,
                                    decode_scale=DECODE_SCALE, **self._seg_kwargs)
        dets = [cv2.resize(np.asarray(m["mask"], np.uint8), (self.pw, self.ph), interpolation=cv2.INTER_NEAREST) > 0
                for m in found]
        self.timings["keyframe_segmentation"] += time.perf_counter() - t0
        return dets

    def _classify_worker(self, crops: List[Image.Image], ids: List[int]):
        t0 = time.perf_counter()
        tax = classify_coral_taxonomy(crops, bioclip_bundle=self.classify["bioclip"], top_k=3)
        ble = detect_coral_bleaching(crops, bleaching_model=self.classify["bleaching"])
        with self._cls_lock:
            self._cls_done.extend((cid, {"taxonomy": t, "bleaching": b}) for cid, t, b in zip(ids, tax, ble))
        self.timings["classification"] += time.perf_counter() - t0

    def _classify_new(self, first_id: int, last_id: int, frame_bgr: np.ndarray):
        """Classify the colonies created by the latest re-anchor, once, from padded full-resolution crops."""
        crops, ids = [], []
        for cid in range(first_id, last_id):
            ys, xs = np.nonzero(self.labels == cid)
            if ys.size < 15:
                continue
            bw, bh = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
            x0, x1 = max(0, xs.min() - 0.1 * bw), min(self.pw, xs.max() + 1 + 0.1 * bw)
            y0, y1 = max(0, ys.min() - 0.1 * bh), min(self.ph, ys.max() + 1 + 0.1 * bh)
            X0, X1 = int(x0 * self.W / self.pw), int(x1 * self.W / self.pw)
            Y0, Y1 = int(y0 * self.H / self.ph), int(y1 * self.H / self.ph)
            if X1 - X0 >= 10 and Y1 - Y0 >= 10:
                crops.append(Image.fromarray(np.ascontiguousarray(frame_bgr[Y0:Y1, X0:X1, ::-1])))
                ids.append(cid)
        if crops:
            self._cls_futs.append(self._cls_pool.submit(self._classify_worker, crops, ids))

    def _apply_keyframe(self, job: Dict[str, Any], frame_bgr: np.ndarray) -> List[int]:
        """Carry a finished keyframe's detections to the current frame through the flows since it started, then re-anchor."""
        dets = job["fut"].result()
        t0 = time.perf_counter()
        det_labels = np.zeros((self.ph, self.pw), np.uint16)
        for i in sorted(range(len(dets)), key=lambda k: -int(dets[k].sum())):
            det_labels[dets[i]] = i + 1
        for flow in job["flows"]:
            det_labels = self._warp(det_labels, flow)
        moved = [m for m in (det_labels == i + 1 for i in range(len(dets))) if m.sum() >= 15]
        first = self.next_id
        self.labels, self.next_id = reanchor(self.labels, moved, self.next_id, self._missed)
        self.timings["reanchor"] += time.perf_counter() - t0
        if self.classify:
            self._classify_new(first, self.next_id, frame_bgr)
        return list(range(first, self.next_id))

    # ---- public API
    def step(self, frame_bgr: np.ndarray) -> FlowStep:
        t0 = time.perf_counter()
        proxy = cv2.resize(frame_bgr, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(proxy, cv2.COLOR_BGR2RGB)
        small = cv2.resize(cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY), (self.fw, self.fh), interpolation=cv2.INTER_AREA)
        self.timings["proxy"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        is_kf, meta = self.selector.evaluate_frame(rgb)
        self.timings["ego_motion"] += time.perf_counter() - t0

        flow = None
        if self._prev_small is not None:
            t0 = time.perf_counter()
            flow = cv2.resize(self._dis.calc(small, self._prev_small, None), (self.pw, self.ph),
                              interpolation=cv2.INTER_LINEAR) / FLOW_SCALE
            self.timings["flow"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            self.labels = self._warp(self.labels, flow)
            self.timings["warp"] += time.perf_counter() - t0
        self._prev_small = small

        applied, new_ids = False, []
        for job in self._jobs:
            if flow is not None:
                job["flows"].append(flow)
        while self._jobs and (self._jobs[0]["fut"].done() or self.n - self._jobs[0]["start"] > MAX_LAG_FRAMES):
            t0 = time.perf_counter()
            self._jobs[0]["fut"].result()
            self.timings["blocked_on_keyframe"] += time.perf_counter() - t0
            new_ids += self._apply_keyframe(self._jobs.pop(0), frame_bgr)
            applied = True

        if self.n == 0:  # the first frame has to be segmented synchronously
            first = self.next_id
            self.labels, self.next_id = reanchor(self.labels, self._segment(rgb, None, None), self.next_id, self._missed)
            new_ids += list(range(first, self.next_id))
            applied = True
            if self.classify:
                self._classify_new(first, self.next_id, frame_bgr)
        elif is_kf:
            if len(self._jobs) < MAX_INFLIGHT:
                pts, mask = guided_prompts(self.labels)
                self._jobs.append({"fut": self._seg_pool.submit(self._segment, rgb, pts, mask), "flows": [], "start": self.n})
            else:
                self.keyframes_skipped += 1

        step = FlowStep(self.n, self.labels, is_kf, applied, float(meta.get("motion_dx", 0.0)),
                        float(meta.get("motion_dy", 0.0)), new_ids)
        self.n += 1
        return step

    def drain_classifications(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Classification results (colony id, {"taxonomy", "bleaching"}) that arrived since the last call."""
        with self._cls_lock:
            done, self._cls_done = self._cls_done, []
        return done

    def finish(self):
        """Stop keyframe work and wait for outstanding classifications (results stay available via drain)."""
        for job in self._jobs:
            job["fut"].cancel()
        self._seg_pool.shutdown(wait=True)
        for fut in self._cls_futs:
            fut.result()
        self._cls_pool.shutdown(wait=True)

    def cancel(self):
        for job in self._jobs:
            job["fut"].cancel()
        self._seg_pool.shutdown(wait=False, cancel_futures=True)
        self._cls_pool.shutdown(wait=False, cancel_futures=True)
