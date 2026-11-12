"""
Parallel Temporal Chunk Scheduler and Colony Identity Reconciliation for ReefVision.
Partitions video into bounded overlapping temporal chunks (e.g. 128 frames with 16-frame overlap),
manages memory-bounded worker admission, executes deterministic track reconciliation across boundaries,
and ensures single-output commit of overlap frames into durable storage.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np

from core.device import check_torch_capabilities, get_memory_budget, can_admit_worker
from core.video.tracker import compute_box_iou


@dataclass
class ChunkRange:
    """Defines a bounded temporal chunk and its identity reconciliation / commit boundaries."""
    chunk_idx: int
    start_frame: int          # inclusive start for decoding and SAM initialization
    end_frame: int            # exclusive end for decoding [start_frame, end_frame)
    reconcile_start: int      # start frame of overlap window for identity matching
    reconcile_end: int        # end frame of overlap window
    commit_start: int         # inclusive start frame for logical output commit
    commit_end: int           # exclusive end frame for logical output commit

    @property
    def total_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def commit_frames_count(self) -> int:
        return self.commit_end - self.commit_start

    @property
    def has_overlap(self) -> bool:
        return self.reconcile_end > self.reconcile_start


def calculate_chunk_plan(
    total_frames: int,
    chunk_size: int = 128,
    overlap_frames: int = 16,
) -> List[ChunkRange]:
    """
    Partitions a video of total_frames into bounded ChunkRange objects.
    Ensures every logical frame in [0, total_frames) is committed exactly once.
    """
    if total_frames <= 0:
        return []

    # If video is smaller than or equal to single chunk size, run in single chunk
    if total_frames <= chunk_size:
        return [
            ChunkRange(
                chunk_idx=0,
                start_frame=0,
                end_frame=total_frames,
                reconcile_start=0,
                reconcile_end=0,
                commit_start=0,
                commit_end=total_frames,
            )
        ]

    chunks = []
    chunk_idx = 0
    cur_start = 0

    while cur_start < total_frames:
        cur_end = min(total_frames, cur_start + chunk_size)

        if chunk_idx == 0:
            reconcile_start = 0
            reconcile_end = 0
            commit_start = 0
            commit_end = cur_end
        else:
            reconcile_start = cur_start
            reconcile_end = cur_start + overlap_frames
            commit_start = cur_start + overlap_frames
            commit_end = cur_end

        chunks.append(
            ChunkRange(
                chunk_idx=chunk_idx,
                start_frame=cur_start,
                end_frame=cur_end,
                reconcile_start=reconcile_start,
                reconcile_end=reconcile_end,
                commit_start=commit_start,
                commit_end=commit_end,
            )
        )

        if cur_end >= total_frames:
            break

        # Next chunk steps forward by (chunk_size - overlap_frames)
        cur_start += (chunk_size - overlap_frames)
        chunk_idx += 1

    return chunks


def compute_pairwise_track_iou(
    track_a: Dict[str, Any],
    track_b: Dict[str, Any],
) -> float:
    """Computes IoU between two track detections on the same frame."""
    box_a = track_a.get("bbox", [0, 0, 0, 0])
    box_b = track_b.get("bbox", [0, 0, 0, 0])
    return compute_box_iou(box_a, box_b)


def reconcile_chunk_identities(
    existing_overlap_frames: Dict[int, List[Dict[str, Any]]],
    incoming_overlap_frames: Dict[int, List[Dict[str, Any]]],
    iou_threshold: float = 0.50,
) -> Dict[int, int]:
    """
    Computes deterministic mean IoU across co-occurring overlap frames between
    an incoming chunk's tracks and existing global colony tracks.
    Returns mapping: {incoming_local_track_id: existing_global_colony_id}.
    Only matches with mean IoU >= iou_threshold are linked.
    """
    co_frames = sorted(set(existing_overlap_frames.keys()) & set(incoming_overlap_frames.keys()))
    if not co_frames:
        return {}

    # Accumulate pairwise IoUs across all co-occurring frames
    pair_ious: Dict[Tuple[int, int], List[float]] = {}

    for f_idx in co_frames:
        existing_tracks = existing_overlap_frames.get(f_idx, [])
        incoming_tracks = incoming_overlap_frames.get(f_idx, [])

        for ex in existing_tracks:
            ex_id = ex.get("global_colony_id", ex.get("track_id"))
            for inc in incoming_tracks:
                inc_id = inc.get("track_id")
                iou = compute_pairwise_track_iou(ex, inc)
                if iou > 0.05:  # filter noise
                    pair_ious.setdefault((ex_id, inc_id), []).append(iou)

    # Compute mean IoU for each candidate pair
    candidate_matches = []
    for (ex_id, inc_id), iou_list in pair_ious.items():
        # Require presence in at least 2 co-occurring frames (or all if overlap is shorter)
        min_presence = min(2, len(co_frames))
        if len(iou_list) >= min_presence:
            mean_iou = sum(iou_list) / len(iou_list)
            if mean_iou >= iou_threshold:
                candidate_matches.append((mean_iou, ex_id, inc_id))

    # Sort descending by mean IoU for greedy optimal matching
    candidate_matches.sort(key=lambda x: x[0], reverse=True)

    matched_existing = set()
    matched_incoming = set()
    reconciliation_map = {}

    for mean_iou, ex_id, inc_id in candidate_matches:
        if ex_id in matched_existing or inc_id in matched_incoming:
            continue
        reconciliation_map[inc_id] = ex_id
        matched_existing.add(ex_id)
        matched_incoming.add(inc_id)

    return reconciliation_map


def admit_workers_for_device(
    device_pref: str,
    requested_workers: int = 1,
) -> int:
    """
    Evaluates accelerator and host memory budget to determine safe worker admission.
    On Apple Silicon (M5 unified memory), defaults to 1 worker to prevent memory thrashing.
    On CUDA, admits up to requested_workers as long as memory budget permits.
    """
    hw_info = check_torch_capabilities()
    is_unified = hw_info.get("is_unified_memory", False)
    device_type = hw_info.get("torch_device", "cpu")

    # On Apple Silicon M5 unified memory, enforce 1 concurrent inference worker
    if is_unified or device_type == "mps":
        return 1

    # On CPU, default to 1 inference worker to prevent thread contention
    if device_type == "cpu" or device_pref == "cpu":
        return 1

    # On CUDA, admit workers based on available VRAM budget
    admitted = 0
    for w in range(1, requested_workers + 1):
        # Estimate ~600 MiB per SAM 2 Tiny worker at 512px
        estimated_vram_bytes = 600 * 1024 * 1024
        estimated_host_bytes = 800 * 1024 * 1024
        can_admit, _ = can_admit_worker(
            estimated_worker_vram_bytes=estimated_vram_bytes,
            estimated_worker_host_bytes=estimated_host_bytes,
        )
        if can_admit:
            admitted += 1
        else:
            break

    return max(1, admitted)
