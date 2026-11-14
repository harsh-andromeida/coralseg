"""
Unit tests for ReefVision Video Processing and Benthic Tracking Pipeline.
Tests camera-motion keyframe selection, durable job storage, exports, hardware reporting and the video API
(the flow tracker itself is covered by test_flow_tracking.py).
"""

import os
import shutil
import tempfile
import time
import numpy as np
import cv2
from PIL import Image

from core.device import check_torch_capabilities, get_memory_budget, resolve_hardware_tier
from core.video.keyframe_selector import EgoMotionKeyframeSelector
from core.video.tracker import CoralTrack
from core.video.video_export import export_transect_csv, export_coco_video, render_video_overlay
from core.video.store import VideoJobStore


def test_video_job_store():
    print("Testing VideoJobStore durable SQLite persistence and LRU cache...")
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_jobs.sqlite3")
    store = VideoJobStore(db_path=db_path, max_cache_frames=4)

    # 1. Create job
    meta = {"width": 1920, "height": 1080, "fps": 25.0, "total_frames": 100}
    opts = {"min_stride": 10, "max_stride": 25}
    job = store.create_job(
        job_id="job_001",
        video_path="/path/test.mp4",
        filename="test.mp4",
        metadata=meta,
        effective_options=opts,
    )
    assert job["job_id"] == "job_001"
    assert job["state"] == "queued"
    assert job["metadata"]["total_frames"] == 100

    # 2. Register chunks & update status
    store.register_chunks("job_001", [(0, 0, 50), (1, 50, 100)])
    chunks = store.get_chunks("job_001")
    assert len(chunks) == 2
    assert chunks[0]["status"] == "pending"

    store.update_chunk_status("job_001", 0, "completed", worker_id="w1")
    completed = store.get_completed_chunk_indices("job_001")
    assert 0 in completed
    assert 1 not in completed

    # 3. Save tracks
    store.save_track(
        job_id="job_001",
        track_id=1,
        global_colony_id=1,
        first_frame=0,
        last_frame=25,
        genus="Acropora",
        condition="Healthy",
    )
    tracks = store.get_tracks("job_001")
    assert 1 in tracks
    assert tracks[1]["genus"] == "Acropora"

    # 4. Save frame batches & LRU caching
    frames_batch = []
    for f in range(10):
        frames_batch.append({
            "frame_idx": f,
            "timestamp_sec": round(f / 25.0, 2),
            "is_keyframe": f == 0,
            "tracks": [{"track_id": 1, "bbox": [10, 10, 20, 20], "area_px": 400}],
            "metrics": {
                "coral_coverage_pct": 5.5,
                "bleaching_prevalence_pct": 0.0,
                "active_corals_count": 1,
            },
        })
    store.save_frames_batch("job_001", frames_batch)

    # Frame retrieval
    f0 = store.get_frame("job_001", 0)
    assert f0 is not None
    assert f0["is_keyframe"] is True
    assert len(f0["tracks"]) == 1

    # Range query
    frange = store.get_frames_range("job_001", 2, 5)
    assert len(frange) == 4
    assert 2 in frange and 5 in frange

    # Delta query
    delta_frames, max_seq = store.get_frames_delta("job_001", since_seq=0, limit=5)
    assert len(delta_frames) == 5
    delta_frames2, max_seq2 = store.get_frames_delta("job_001", since_seq=max_seq, limit=10)
    assert len(delta_frames2) == 5
    assert max_seq2 > max_seq

    # Fast time-series query
    ts = store.get_time_series("job_001")
    assert len(ts) == 10
    assert ts[0]["coral_coverage_pct"] == 5.5

    store.close()
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("  VideoJobStore verified successfully.")


def test_device_capabilities_and_precision():
    print("Testing device capabilities and precision detection...")
    caps = check_torch_capabilities()
    assert "torch_available" in caps
    assert "native_precision" in caps
    native = caps["native_precision"]
    assert "cuda_bf16_native" in native
    assert "mps_fp32" in native

    budget = get_memory_budget()
    assert "host_total_bytes" in budget
    assert "host_available_bytes" in budget
    assert "min_device_reservation_bytes" in budget
    assert "min_host_reservation_bytes" in budget
    assert budget["min_host_reservation_bytes"] >= 2 * 1024**3
    print("  Hardware introspection and memory budget verified.")


def test_keyframe_selector():
    print("Testing EgoMotionKeyframeSelector...")
    selector = EgoMotionKeyframeSelector(
        overlap_threshold=0.75,
        min_stride=3,
        max_stride=10,
    )

    # Frame 1: must be a keyframe
    f1 = np.full((100, 100, 3), 120, dtype=np.uint8)
    cv2.circle(f1, (50, 50), 20, (200, 50, 50), -1)
    is_kf, meta = selector.evaluate_frame(f1)
    assert is_kf is True, "First frame must be keyframe"
    assert meta["keyframe_index"] == 1

    # Frame 2: identical frame -> tracking frame
    is_kf, meta = selector.evaluate_frame(f1)
    assert is_kf is False, "Identical frame should not trigger keyframe"
    assert meta["overlap_ratio"] > 0.90

    # Frame 3: small motion
    f3 = np.roll(f1, shift=2, axis=1)
    is_kf, meta = selector.evaluate_frame(f3)
    assert is_kf is False

    # Force max stride keyframe (max_stride=10)
    triggered_at_stride = False
    for _ in range(12):
        is_kf, meta = selector.evaluate_frame(f1)
        if is_kf:
            triggered_at_stride = True
            break
    assert triggered_at_stride is True, "Max stride must trigger safety keyframe"
    print("  Keyframe selector passed.")


def test_export_utilities():
    print("Testing Video Export utilities...")
    temp_dir = tempfile.mkdtemp(prefix="reefvision_test_vid_")
    try:
        csv_path = os.path.join(temp_dir, "transect.csv")
        records = [{
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "active_corals_count": 2,
            "cumulative_unique_corals": 2,
            "bleached_corals_count": 0,
            "healthy_corals_count": 2,
            "bleaching_prevalence_pct": 0.0,
            "dominant_genus": "Acropora",
            "is_keyframe": True,
            "camera_motion_dx": 0.0,
            "camera_motion_dy": 0.0,
        }]
        export_transect_csv(csv_path, records)
        assert os.path.exists(csv_path)

        coco_path = os.path.join(temp_dir, "coco_video.json")
        export_coco_video(
            output_json_path=coco_path,
            video_filename="test.mp4",
            width=100,
            height=100,
            total_frames=1,
            fps=30.0,
            all_tracks=[],
            frame_tracks_map={},
        )
        assert os.path.exists(coco_path)
        print("  Export utilities passed.")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_video_api_endpoints():
    print("Testing Video FastAPI Endpoints & Rejection Gates...")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 2. Test rejection of invalid strides
    resp_stride = client.post("/api/video/process", json={
        "video_name": "non_existent.mp4",
        "min_stride": 50,
        "max_stride": 20,
    })
    assert resp_stride.status_code == 422, f"Expected 422 for min_stride > max_stride, got {resp_stride.status_code}"
    print("  Rejected invalid min_stride > max_stride verified with 422.")

    # 3. Create a synthetic 5-frame MP4 to test execution lifecycle
    temp_dir = tempfile.mkdtemp(prefix="test_mp4_")
    test_vid_path = os.path.join(temp_dir, "test_synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(test_vid_path, fourcc, 10.0, (120, 120))
    for i in range(5):
        frame = np.full((120, 120, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (30 + i * 4, 60), 18, (180, 80, 50), -1)
        writer.write(frame)
    writer.release()

    try:
        # 4. Upload video
        with open(test_vid_path, "rb") as f:
            resp = client.post("/api/video/upload", files={"file": ("test_synthetic.mp4", f, "video/mp4")})
        assert resp.status_code == 200, f"Video upload failed: {resp.text}"
        upload_data = resp.json()
        assert upload_data["status"] == "success"
        assert upload_data["metadata"]["fps"] == 25.0
        assert upload_data["metadata"]["total_frames"] > 0
        print("  POST /api/video/upload verified (standardized to 25 FPS).")

        # 5. Process video
        resp = client.post("/api/video/process", json={
            "video_name": upload_data["video_name"],
            "overlap_threshold": 0.75,
            "min_stride": 2,
            "max_stride": 5,
        })
        assert resp.status_code == 200, f"Process video failed: {resp.text}"
        job_data = resp.json()
        job_id = job_data["job_id"]
        print(f"  POST /api/video/process verified. Job ID: {job_id}")

        # 6. Status polling (wait for completion)
        completed = False
        s_data = {}
        for _ in range(250):
            time.sleep(0.5)
            s_resp = client.get(f"/api/video/status/{job_id}")
            assert s_resp.status_code == 200
            s_data = s_resp.json()
            if s_data["state"] in ("completed", "error"):
                completed = True
                break

        assert completed is True, f"Job did not finish in time: {s_data}"
        assert s_data["state"] == "completed", f"Job failed: {s_data.get('error')}"
        assert s_data["current_frame"] == s_data["total_frames"]
        assert "tracker_fps" in s_data
        assert "analysis_fps" in s_data
        print(f"  GET /api/video/status verified: completed {s_data['current_frame']} frames at {s_data['analysis_fps']} FPS.")

        # 7. Frame annotations
        f_resp = client.get(f"/api/video/frame/{job_id}/0")
        assert f_resp.status_code == 200
        f_data = f_resp.json()
        assert "tracks" in f_data
        print(f"  GET /api/video/frame/0 verified: {f_data['active_count']} active tracks.")

        # 8. Bulk frame annotations
        bulk_resp = client.get(f"/api/video/frames/{job_id}?start_frame=0&end_frame=1")
        assert bulk_resp.status_code == 200
        bulk_data = bulk_resp.json()
        assert "frames" in bulk_data
        print(f"  GET /api/video/frames verified: {bulk_data['total_frames_available']} frames in bulk.")

        # 9. Delta polling endpoint
        delta_resp = client.get(f"/api/video/delta/{job_id}?since_seq=0&limit=5")
        assert delta_resp.status_code == 200
        delta_data = delta_resp.json()
        assert "frames" in delta_data
        print(f"  GET /api/video/delta verified: {len(delta_data['frames'])} frames polled.")

        # 10. Export CSV
        exp_resp = client.get(f"/api/video/export/{job_id}?format=csv")
        assert exp_resp.status_code == 200
        assert len(exp_resp.content) > 0
        print("  GET /api/video/export (CSV) verified.")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_real_video_sample_pipeline():
    """Verifies end-to-end flow tracking, keyframing, and export on real underwater reef footage."""
    print("\n--- Testing Real Underwater Video Sample Pipeline ---")
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    # 1. Fetch sample video
    resp = client.get("/api/video/sample")
    assert resp.status_code == 200, f"Sample video fetch failed: {resp.text}"
    sample_data = resp.json()
    video_name = sample_data["video_name"]
    meta = sample_data["metadata"]
    print(f"  Loaded sample video '{video_name}': {meta['width']}x{meta['height']} @ {meta['fps']}fps, {meta['total_frames']} frames")
    assert meta["width"] > 0 and meta["height"] > 0
    assert meta["fps"] == 25.0
    assert meta["total_frames"] > 0

    # 2. Launch processing job
    p_resp = client.post("/api/video/process", json={
        "video_name": video_name,
        "overlap_threshold": 0.70,
        "min_stride": 15,
        "max_stride": 45,
        "points_per_side": 8,
    })
    assert p_resp.status_code == 200, f"Start job failed: {p_resp.text}"
    job_id = p_resp.json()["job_id"]
    print(f"  Launched processing job '{job_id}'")

    # 3. Poll until completion
    s_data = {}
    for poll_idx in range(600):
        time.sleep(1.0)
        s_resp = client.get(f"/api/video/status/{job_id}")
        assert s_resp.status_code == 200
        s_data = s_resp.json()
        if poll_idx % 15 == 0:
            print(f"    Progress: frame {s_data.get('current_frame', 0)}/{s_data.get('total_frames', 0)} ({s_data.get('progress_pct', 0.0):.1f}%) | state={s_data.get('state')} | analysis_fps={s_data.get('analysis_fps', 0.0)}")
        if s_data["state"] in ("completed", "error"):
            break

    assert s_data["state"] == "completed", f"Job failed: {s_data.get('error')}"
    print(f"  Completed! Processed {s_data['current_frame']} frames | Keyframes: {s_data['keyframes_count']} | Total corals: {s_data['total_unique_corals']} | Tracker FPS: {s_data.get('tracker_fps', 0.0)}")
    assert s_data["keyframes_count"] >= 1
    assert s_data["total_unique_corals"] >= 1

    # 4. Verify frame annotations
    frame_resp = client.get(f"/api/video/frame/{job_id}/0")
    assert frame_resp.status_code == 200
    f_data = frame_resp.json()
    assert "tracks" in f_data
    print(f"  Frame 0 annotations: {len(f_data['tracks'])} tracks detected")

    # 5. Verify exports (MP4, CSV, JSON)
    for fmt in ("mp4", "csv", "json"):
        exp_resp = client.get(f"/api/video/export/{job_id}?format={fmt}")
        assert exp_resp.status_code == 200, f"Export format '{fmt}' failed"
        assert len(exp_resp.content) > 0
        print(f"  Export '{fmt}' verified ({len(exp_resp.content)} bytes)")


if __name__ == "__main__":
    test_video_job_store()
    test_keyframe_selector()
    test_device_capabilities_and_precision()
    test_export_utilities()
    test_video_api_endpoints()
    test_real_video_sample_pipeline()
    print("\nAll video pipeline unit tests PASSED successfully!")
