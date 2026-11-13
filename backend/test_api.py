#!/usr/bin/env python3
"""
Test script for ReefVision FastAPI endpoints using FastAPI TestClient.
"""

import sys
import io
import base64
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app


def test_endpoints():
    client = TestClient(app)

    print("Testing /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health failed: {resp.text}"
    health = resp.json()
    print(f"  Status: {health['status']}, active device: {health['device']['device']}")

    print("Testing /api/models/status...")
    resp = client.get("/api/models/status")
    assert resp.status_code == 200, f"Models status failed: {resp.text}"
    models = resp.json()
    print(f"  All models downloaded: {models['all_downloaded']}, model count: {len(models['models'])}")

    print("Testing /api/samples...")
    resp = client.get("/api/samples")
    assert resp.status_code == 200, f"Samples failed: {resp.text}"
    samples = resp.json()
    print(f"  Found {len(samples)} samples.")
    if samples:
        print(f"  First sample: {samples[0]['filename']}")

    print("Testing /api/device/select across hardware priority tiers...")
    # Tier 1: GPU
    resp = client.post("/api/device/select", json={"preference": "gpu"})
    assert resp.status_code == 200, f"Device select (gpu) failed: {resp.text}"
    dev_gpu = resp.json()
    print(f"  GPU select -> mode: {dev_gpu.get('mode')}, provider: {dev_gpu.get('device')}, label: {dev_gpu.get('mode_label')}")

    # Tier 2: Multi-Threaded CPU
    resp = client.post("/api/device/select", json={"preference": "multithread_cpu"})
    assert resp.status_code == 200, f"Device select (multithread_cpu) failed: {resp.text}"
    dev_mt = resp.json()
    assert dev_mt.get("mode") == "multithread_cpu", f"Expected multithread_cpu, got {dev_mt.get('mode')}"
    print(f"  Multi-thread CPU select -> mode: {dev_mt.get('mode')}, threads: {dev_mt.get('threads')}, label: {dev_mt.get('gpu_name')}")

    # Tier 3: Single-Threaded CPU
    resp = client.post("/api/device/select", json={"preference": "cpu"})
    assert resp.status_code == 200, f"Device select (cpu) failed: {resp.text}"
    dev_cpu = resp.json()
    assert dev_cpu.get("mode") == "cpu", f"Expected cpu, got {dev_cpu.get('mode')}"
    assert dev_cpu.get("threads") == 1, f"Expected 1 thread for single-threaded CPU, got {dev_cpu.get('threads')}"
    print(f"  Single-thread CPU select -> mode: {dev_cpu.get('mode')}, threads: {dev_cpu.get('threads')}, label: {dev_cpu.get('gpu_name')}")

    # Auto mode resolution (GPU -> Multi-threaded CPU -> CPU)
    resp = client.post("/api/device/select", json={"preference": "auto"})
    assert resp.status_code == 200, f"Device select (auto) failed: {resp.text}"
    dev_auto = resp.json()
    print(f"  Auto select -> mode: {dev_auto.get('mode')}, priority: {dev_auto.get('priority')}, active: {dev_auto.get('gpu_name')}")

    print("Testing /api/models/delete endpoint structure...")
    # Safe test with empty target so we don't delete local weights during basic unit test
    resp = client.post("/api/models/delete", json={"filename": "unknown_file.onnx"})
    assert resp.status_code == 200, f"Models delete failed: {resp.text}"
    del_res = resp.json()
    assert "all_downloaded" in del_res, "Response missing all_downloaded"
    assert "models" in del_res, "Response missing models list"
    assert del_res["status"] == "success", "Expected status success"
    print(f"  Delete endpoint response verified: status={del_res['status']}")

    print("Testing constraint enforcement: segment & enrich blocked when models missing...")
    import app.main as app_main
    orig_check = app_main.check_models_download_status
    try:
        # Mock models as not downloaded
        app_main.check_models_download_status = lambda: (False, [])
        resp = client.post("/api/analysis/segment", json={"image_name": "test_img.png"})
        assert resp.status_code == 428, f"Expected 428 Precondition Required, got {resp.status_code}: {resp.text}"
        print(f"  /api/analysis/segment correctly blocked with 428: {resp.json().get('detail')}")

        resp = client.post("/api/analysis/enrich", json={"image_name": "test_img.png", "min_area_px": 100})
        assert resp.status_code == 428, f"Expected 428 Precondition Required, got {resp.status_code}: {resp.text}"
        print(f"  /api/analysis/enrich correctly blocked with 428: {resp.json().get('detail')}")
    finally:
        app_main.check_models_download_status = orig_check

    print("Testing /api/batch endpoints lifecycle...")
    # Status
    resp = client.get("/api/batch/status")
    assert resp.status_code == 200, f"Batch status failed: {resp.text}"
    b_status = resp.json()
    assert "is_running" in b_status
    print(f"  Initial batch status: running={b_status['is_running']}, total={b_status['total']}")

    # Pause / Resume / Cancel
    resp = client.post("/api/batch/pause")
    assert resp.status_code == 200
    resp = client.post("/api/batch/resume")
    assert resp.status_code == 200
    resp = client.post("/api/batch/cancel")
    assert resp.status_code == 200

    # Test /api/models/unload endpoint
    resp = client.post("/api/models/unload")
    assert resp.status_code == 200
    assert resp.json().get("unloaded") is True

    # Prioritize
    resp = client.post("/api/batch/prioritize", json={"image_name": "sample.png"})
    assert resp.status_code == 200
    print("  Batch pause, resume, cancel, unload, prioritize endpoints verified.")

    # Export empty -> 400
    resp = client.get("/api/batch/export/coco-zip")
    assert resp.status_code in (400, 200)

    # Test COCO dataset ZIP serialization directly
    import io, zipfile
    from PIL import Image
    from core.image_export import create_coco_dataset_zip
    mock_img = Image.new("RGB", (64, 64), color=(0, 100, 200))
    mock_store = {"coral_test.png": mock_img}
    mock_item = {
        "image_name": "coral_test.png",
        "width": 64,
        "height": 64,
        "masks_info": [],
        "segments": [{
            "id": 1,
            "id_str": "#1",
            "genus": "Acropora",
            "growth_form": "Branching",
            "taxon_conf": 95.0,
            "condition": "Healthy",
            "condition_conf": 90.0,
            "area_pct": 12.5,
            "area_px": 512,
            "predicted_iou": 0.94,
            "bbox": [10, 10, 20, 20],
            "centroid": [20, 20],
        }],
        "corals_count": 1,
        "coverage_pct": 12.5,
        "bleaching_prevalence_pct": 0.0,
        "summary": {},
        "health_summary": {},
        "scene_eval": {},
    }
    zip_bytes = create_coco_dataset_zip([mock_item], mock_store)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    namelist = zf.namelist()
    assert "annotations/instances_default.json" in namelist
    assert "summary.csv" in namelist
    assert "dataset_summary.json" in namelist
    assert "images/coral_test.png" in namelist
    print(f"  COCO dataset zip verified with files: {namelist}")

    # Test GET /api/batch/export/default-path
    resp = client.get("/api/batch/export/default-path")
    assert resp.status_code == 200, f"Default path failed: {resp.text}"
    def_path_info = resp.json()
    assert "default_folder" in def_path_info
    print(f"  Export default-path verified: {def_path_info['default_folder']}")

    # Test POST /api/batch/export/to-folder with mock item and underscore filename
    import tempfile, shutil
    from app.state import _IMAGE_STORE, _ENRICH_CACHE
    test_img_name = "test_coral_reef_01.png"
    _IMAGE_STORE[test_img_name] = mock_img
    _ENRICH_CACHE[f"{test_img_name}_16_0.5_0.5_100"] = (
        [{
            "id": 1,
            "area_px": 512,
            "area_pct": 12.5,
            "predicted_iou": 0.94,
            "taxonomy": {"genus": "Acropora", "growth_form": "Branching", "confidence_pct": 95.0},
            "bleaching": {"is_bleached": False, "confidence_pct": 90.0},
            "color_hex": "#00cccc",
            "bbox": [10, 10, 20, 20],
            "centroid": [20, 20],
        }],
        {"overall_assessment": "healthy"},
        {"bleaching_prevalence_pct": 0.0},
    )

    # Test export to zip with underscore image
    resp = client.get(f"/api/batch/export/coco-zip?image_name={test_img_name}")
    assert resp.status_code == 200, f"Underscore image zip export failed: {resp.text}"
    assert len(resp.content) > 0
    print("  COCO dataset zip for image with underscores verified.")

    # Test export to custom directory
    temp_export_dir = tempfile.mkdtemp(prefix="reefvision_test_export_")
    try:
        resp = client.post("/api/batch/export/to-folder", json={
            "folder_path": temp_export_dir,
            "include_zip": True,
            "open_folder": False,
        })
        assert resp.status_code == 200, f"Export to folder failed: {resp.text}"
        res_data = resp.json()
        assert res_data["success"] is True
        assert res_data["total_images"] >= 1
        export_p = Path(temp_export_dir)
        assert (export_p / "annotations" / "instances_default.json").exists()
        assert (export_p / "images" / test_img_name).exists()
        assert (export_p / "summary.csv").exists()
        assert (export_p / "dataset_summary.json").exists()
        assert (export_p / "README.md").exists()
        print(f"  Export to custom folder verified with {res_data['written_files_count']} files.")
    finally:
        shutil.rmtree(temp_export_dir, ignore_errors=True)
        _IMAGE_STORE.pop(test_img_name, None)
        _ENRICH_CACHE.pop(f"{test_img_name}_16_0.5_0.5_100", None)

    print("Testing manual annotation endpoints...")
    # Register test image for manual annotation
    manual_img_name = "manual_test_coral.png"
    pil_test = Image.new("RGB", (200, 200), color=(10, 40, 60))
    _IMAGE_STORE[manual_img_name] = pil_test

    # 1. Test get polygons when empty
    resp = client.get(f"/api/analysis/polygons/{manual_img_name}")
    assert resp.status_code == 200, f"Get polygons failed: {resp.text}"
    poly_data = resp.json()
    assert poly_data["image_name"] == manual_img_name
    assert poly_data["width"] == 200
    assert poly_data["height"] == 200
    print("  Polygons retrieval endpoint verified.")

    # 2. Test SAM interactive point prompt segmentation
    resp = client.post("/api/analysis/segment-prompt", json={
        "image_name": manual_img_name,
        "points": [{"x": 100.0, "y": 100.0, "label": 1}],
    })
    assert resp.status_code == 200, f"SAM prompt segment failed: {resp.text}"
    prompt_res = resp.json()
    assert prompt_res["status"] == "success"
    assert "polygon" in prompt_res
    assert "area_px" in prompt_res
    print(f"  Interactive SAM prompt segmentation verified: area_px={prompt_res['area_px']}, vertices={len(prompt_res['polygon'])}")

    # 2b. Test SAM interactive segmentation on un-cached image with image_data dataUrl
    b64_buf = io.BytesIO()
    pil_test.save(b64_buf, format="PNG")
    raw_b64 = base64.b64encode(b64_buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{raw_b64}"
    resp_b64 = client.post("/api/analysis/segment-prompt", json={
        "image_name": "custom_uncached_coral.png",
        "points": [{"x": 50.0, "y": 50.0, "label": 1}],
        "image_data": data_url,
    })
    assert resp_b64.status_code == 200, f"SAM prompt with image_data failed: {resp_b64.text}"
    print("  SAM prompt segmentation with raw image_data verified.")

    # 3. Test Save Annotations with Ground Truth
    sample_polygon = [[40.0, 40.0], [120.0, 40.0], [120.0, 120.0], [40.0, 120.0], [40.0, 40.0]]
    resp = client.post("/api/analysis/save-annotations", json={
        "image_name": manual_img_name,
        "segments": [
            {
                "id": 1,
                "genus": "Acropora",
                "growth_form": "Branching",
                "condition": "Healthy",
                "color_hex": "#da70d6",
                "polygon": sample_polygon,
            },
            {
                "id": 2,
                "genus": "Porites",
                "growth_form": "Massive",
                "condition": "Bleached",
                "color_hex": "#ffcc00",
                "polygon": [[130.0, 130.0], [170.0, 130.0], [170.0, 170.0], [130.0, 170.0], [130.0, 130.0]],
            },
        ],
    })
    assert resp.status_code == 200, f"Save annotations failed: {resp.text}"
    save_res = resp.json()
    assert save_res["status"] == "success"
    assert len(save_res["segments"]) == 2
    # Verify 100% human ground truth confidence
    for s in save_res["segments"]:
        assert s["taxon_conf"] == 100.0, f"Expected 100% taxon confidence, got {s['taxon_conf']}"
        assert s["condition_conf"] == 100.0, f"Expected 100% condition confidence, got {s['condition_conf']}"
        assert s["predicted_iou"] == 1.0, f"Expected 1.0 IoU, got {s['predicted_iou']}"
        assert s["is_human_annotated"] is True
    # Verify area recalculated
    assert save_res["summary"]["coral_coverage_pct"] > 0
    assert save_res["summary"]["mean_iou_confidence"] == 1.0
    assert save_res["health_summary"]["healthy_count"] == 1
    assert save_res["health_summary"]["bleached_count"] == 1
    assert save_res["overlay_base64"].startswith("data:image/png;base64,")
    print(f"  Save annotations verified: 100% ground truth confidences, coverage={save_res['summary']['coral_coverage_pct']}%, overlay rendered.")

    # Cleanup
    _IMAGE_STORE.pop(manual_img_name, None)

    print("\nAll API endpoint unit tests PASSED successfully!")



if __name__ == "__main__":
    test_endpoints()
