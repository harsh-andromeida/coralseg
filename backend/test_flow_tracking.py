#!/usr/bin/env python3
"""Fast, model-free unit tests for the label bookkeeping in core/video/flow_tracking.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from core.video import flow_tracking as ft


def _mask(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_reanchor_creates_ids_then_reuses_them():
    shape = (60, 80)
    labels = np.zeros(shape, np.uint16)
    missed = {}
    labels, next_id = ft.reanchor(labels, [_mask(shape, 5, 25, 5, 25), _mask(shape, 30, 55, 40, 70)], 1, missed)
    assert next_id == 3 and set(np.unique(labels)) == {0, 1, 2}

    # the same colonies seen again, slightly shifted: ids are reused, no new colony is created
    shifted = [_mask(shape, 7, 27, 6, 26), _mask(shape, 31, 56, 41, 71)]
    ious = []
    labels, next_id = ft.reanchor(labels, shifted, next_id, missed, ious)
    assert next_id == 3 and set(np.unique(labels)) == {0, 1, 2} and len(ious) == 2 and min(ious) > 0.6


def test_reanchor_matches_sub_part_by_containment():
    shape = (60, 80)
    labels = np.zeros(shape, np.uint16)
    labels, next_id = ft.reanchor(labels, [_mask(shape, 5, 45, 5, 45)], 1, {})
    # CoralSCOP returns only a small part of the same colony: IoU is low but containment is high
    labels, next_id = ft.reanchor(labels, [_mask(shape, 10, 25, 10, 25)], next_id, {})
    assert next_id == 2


def test_reanchor_keeps_missed_colony_then_drops_it():
    shape = (60, 80)
    missed = {}
    labels, next_id = ft.reanchor(np.zeros(shape, np.uint16), [_mask(shape, 5, 25, 5, 25)], 1, missed)
    for _ in range(ft.MAX_MISSED_KEYFRAMES - 1):  # not re-detected: still present
        labels, next_id = ft.reanchor(labels, [_mask(shape, 35, 55, 45, 70)], next_id, missed)
        assert 1 in np.unique(labels)
    labels, next_id = ft.reanchor(labels, [_mask(shape, 35, 55, 45, 70)], next_id, missed)
    assert 1 not in np.unique(labels)


def test_guided_prompts_skip_known_area():
    labels = np.zeros((60, 80), np.uint16)
    labels[10:40, 10:50] = 1
    pts, free = ft.guided_prompts(labels)
    assert pts.shape == (1, 2) and 10 <= pts[0, 0] < 50 and 10 <= pts[0, 1] < 40
    assert free[int(pts[0, 1]), int(pts[0, 0])]   # the colony's own prompt point is allowed
    assert free[55, 70] and not free[12, 12]       # open reef is prompted, known colony area is not
    pts0, free0 = ft.guided_prompts(np.zeros((60, 80), np.uint16))
    assert pts0 is None and free0 is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All flow_tracking tests PASSED")
