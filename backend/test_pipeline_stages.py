#!/usr/bin/env python3
"""Fast, model-free unit tests for core/video/pipeline_stages.py."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

from core.video import pipeline_stages as ps


def test_rescale_detections():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 3:7] = True
    dets = [{"mask": mask, "bbox": [3, 2, 4, 4], "centroid": (5, 4), "area": 16, "area_px": 16, "polygon": [[3, 2], [6, 2], [6, 5]]}]
    ps.rescale_detections_to_full_res(dets, 2.0, 3.0)
    d = dets[0]
    assert d["bbox"] == [6, 6, 8, 12]
    assert d["centroid"] == (10, 12)
    assert d["area"] == 96 and d["area_px"] == 96
    assert d["polygon"] == [[6, 6], [12, 6], [12, 15]]


def test_rescale_derives_missing_polygon():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:14, 4:14] = True
    dets = [{"mask": mask, "bbox": [4, 4, 10, 10]}]
    ps.rescale_detections_to_full_res(dets, 2.0, 2.0)
    assert dets[0]["polygon"], "polygon should be extracted then scaled"
    assert max(p[0] for p in dets[0]["polygon"]) >= 18


def _trk(**kw):
    base = dict(track_id=7, taxonomy={}, bleaching={}, color_hex="#fff", first_seen_frame=3, predicted_iou=0.9, bbox=[10, 10, 50, 40])
    base.update(kw)
    return SimpleNamespace(**base)


def test_track_metadata_defaults_and_override():
    m = ps.track_metadata(_trk())
    assert m["genus"] == "Coral" and m["condition"] == "Healthy" and m["taxon_conf"] == 85.0
    assert m["first_seen_frame"] == 3
    assert ps.track_metadata(_trk(), 11)["first_seen_frame"] == 11
    m2 = ps.track_metadata(_trk(taxonomy={"genus": "Acropora"}, bleaching={"status": "Bleached", "is_bleached": True}))
    assert m2["genus"] == "Acropora" and m2["is_bleached"] is True


def test_crop_track_regions_padding_and_tiny_boxes():
    img = Image.new("RGB", (200, 100))
    crops = ps.crop_track_regions(img, [_trk(bbox=[10, 10, 50, 40]), _trk(bbox=[0, 0, 5, 5])], 200, 100)
    assert crops[0].size == (60, 48)  # 10% padding each side
    assert crops[1].size == (10, 10)  # minimum size fallback


def test_build_pipeline_timings():
    t = ps.build_pipeline_timings(
        elapsed_total=100.0, compile_warmup_sec=1, initialization_sec=2, motion_scan_sec=5, chunk_prep_sec=5,
        propagation_sec=50, transfer_sec=5, geometry_sec=5, queue_wait_sec=0, discovery_sec=10,
        classification_sec=3, persistence_sec=2, export_deliverables_sec=1, export_mp4_sec=7,
        unique_colonies=4, keyframes_count=9, committed_frames=202,
    )
    assert t["video_decode_prep_sec"] == 10.0
    assert t["unaccounted_idle_sec"] == 10.0  # 100 - (10+50+5+5+10+2+1+7)
    assert t["pct_gpu_inference"] == 50.0 and t["pct_export"] == 8.0
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
