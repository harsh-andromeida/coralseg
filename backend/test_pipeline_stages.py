#!/usr/bin/env python3
"""Fast, model-free unit tests for core/video/pipeline_stages.py."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from core.video import pipeline_stages as ps


def _blob(labels, cid, y0, y1, x0, x1):
    labels[y0:y1, x0:x1] = cid


def test_labels_to_geometry_scales_and_skips_tiny():
    labels = np.zeros((20, 40), dtype=np.uint16)
    _blob(labels, 3, 2, 12, 4, 14)    # 10x10 colony
    _blob(labels, 5, 15, 16, 30, 32)  # 2 px speck, below min_pixels
    geo = ps.labels_to_geometry(labels, width=160, height=80)  # 4x scale in both axes
    assert sorted(geo) == [3]
    g = geo[3]
    assert g["bbox"] == [16, 8, 40, 40]
    assert g["area_px"] == 100 * 16 and g["area_pct"] == round(1600 / (160 * 80) * 100.0, 4)
    assert g["centroid"] == (int(round(8.5 * 4)), int(round(6.5 * 4)))
    xs = [p[0] for p in g["polygon"]]
    ys = [p[1] for p in g["polygon"]]
    assert min(xs) == 16 and max(xs) == 52 and min(ys) == 8 and max(ys) == 44


def test_labels_to_geometry_empty_and_multiple():
    assert ps.labels_to_geometry(np.zeros((10, 10), np.uint16), 100, 100) == {}
    labels = np.zeros((30, 30), dtype=np.uint16)
    _blob(labels, 1, 0, 8, 0, 8)
    _blob(labels, 7, 10, 25, 10, 25)
    geo = ps.labels_to_geometry(labels, 30, 30)
    assert sorted(geo) == [1, 7] and geo[7]["area_px"] == 225


def test_colony_color_is_stable_and_distinct():
    assert ps.colony_color(4) == ps.colony_color(4)
    assert len({ps.colony_color(i) for i in range(1, 30)}) == 29


def test_build_pipeline_timings():
    t = ps.build_pipeline_timings(
        elapsed_total=100.0, model_load_sec=2, decode_sec=10, ego_motion_sec=5, flow_sec=40, geometry_sec=10,
        keyframe_blocked_sec=3, keyframe_segmentation_sec=30, classification_sec=3, persistence_sec=2,
        export_deliverables_sec=1, export_mp4_sec=7, unique_colonies=4, keyframes_count=9, committed_frames=202,
    )
    assert t["unaccounted_idle_sec"] == 22.0  # 100 - (10+5+40+10+3+2+1+7); background threads are not on the path
    assert t["pct_optical_flow_and_warp"] == 40.0 and t["pct_export"] == 8.0
    assert t["committed_frames"] == 202 and t["unique_colonies_count"] == 4


def test_run_deliverable_exports_statuses(monkeypatch=None):
    logs = []
    set_stage = lambda **kw: logs.append(kw)
    orig = (ps.export_transect_csv, ps.export_coco_video, ps.export_annotated_video)
    try:
        def ok_csv(output_csv_path, time_series_records): Path(output_csv_path).write_text("a,b\n1,2\n")
        def ok_coco(output_json_path, **kw): Path(output_json_path).write_text("{}")
        def bad_mp4(**kw): raise RuntimeError("no ffmpeg")
        ps.export_transect_csv, ps.export_coco_video, ps.export_annotated_video = ok_csv, ok_coco, bad_mp4
        with tempfile.TemporaryDirectory() as d:
            common = dict(cache_dir=d, job_id="j", filename="v.mp4", video_path="v.mp4", width=1, height=1,
                          total_frames=1, fps=25.0, tracks=[], time_series_data=[], frame_adapter=None, set_stage=set_stage)
            r = ps.run_deliverable_exports(export_mp4=True, **common)
            assert r.status == "partial" and "annotated_mp4" in r.errors and r.paths["transect_csv"]
            r = ps.run_deliverable_exports(export_mp4=False, **common)
            assert r.status == "ready" and r.mp4_sec == 0.0
            ps.export_transect_csv = lambda **kw: None  # writes nothing -> "0 bytes" error
            os.remove(os.path.join(d, "j_transect.csv"))
            r = ps.run_deliverable_exports(export_mp4=False, **common)
            assert r.status == "partial" and "transect_csv" in r.errors
    finally:
        ps.export_transect_csv, ps.export_coco_video, ps.export_annotated_video = orig


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All pipeline_stages tests PASSED")
