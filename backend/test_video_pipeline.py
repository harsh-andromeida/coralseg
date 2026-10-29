"""
Unit tests for ReefVision High-FPS Video Processing and Benthic Tracking Pipeline.
"""

import os
import tempfile
import numpy as np
import cv2
from PIL import Image

from core.video.keyframe_selector import EgoMotionKeyframeSelector
from core.video.tracker import BenthicColonyTracker, CoralTrack, compute_box_iou
from core.video.export import export_transect_csv, export_coco_video, render_video_overlay


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


def test_benthic_tracker():
    print("Testing BenthicColonyTracker...")
    tracker = BenthicColonyTracker(iou_match_threshold=0.30)

    # Keyframe 1: 2 coral detections
    dets_f1 = [
        {
            "bbox": [10, 10, 30, 30],
            "polygon": [[10, 10], [40, 10], [40, 40], [10, 40]],
            "area_px": 900,
            "area_pct": 5.0,
            "predicted_iou": 0.92,
            "centroid": (25, 25),
        },
        {
            "bbox": [60, 60, 20, 20],
            "polygon": [[60, 60], [80, 60], [80, 80], [60, 80]],
            "area_px": 400,
            "area_pct": 2.5,
            "predicted_iou": 0.88,
            "centroid": (70, 70),
        },
    ]

    tracks_f1 = tracker.match_keyframe_detections(dets_f1, frame_idx=0, frame_w=100, frame_h=100)
    assert len(tracks_f1) == 2
    assert tracks_f1[0].track_id == 1
    assert tracks_f1[1].track_id == 2
    print(f"  Spawned {len(tracks_f1)} tracks on frame 0.")

    # Intervening frame: camera shifts by dx=2, dy=1
    step_H = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    tracker.warp_tracks_with_motion(step_H, frame_w=100, frame_h=100, frame_idx=1)
    tracks_f2 = tracker.get_active_tracks()
    assert len(tracks_f2) == 2
    assert tracks_f2[0].track_id == 1
    # Check warped bbox shifted
    assert tracks_f2[0].bbox[0] >= 11, f"Expected x shifted from 10, got {tracks_f2[0].bbox[0]}"
    print("  Inter-frame homography warping verified.")

    # Keyframe 2: Refined detections matched with persistent IDs
    dets_f2 = [
        {
            "bbox": [12, 11, 30, 30],
            "polygon": [[12, 11], [42, 11], [42, 41], [12, 41]],
            "area_px": 900,
            "area_pct": 5.0,
            "predicted_iou": 0.94,
            "centroid": (27, 26),
        },
        {
            "bbox": [62, 61, 20, 20],
            "polygon": [[62, 61], [82, 61], [82, 81], [62, 81]],
            "area_px": 400,
            "area_pct": 2.5,
            "predicted_iou": 0.90,
            "centroid": (72, 71),
        },
    ]

    tracks_f3 = tracker.match_keyframe_detections(dets_f2, frame_idx=2, frame_w=100, frame_h=100)
    assert len(tracks_f3) == 2
    assert tracks_f3[0].track_id == 1, "Track ID 1 must persist"
    assert tracks_f3[1].track_id == 2, "Track ID 2 must persist"
    print("  Track persistence & Hungarian IoU matching verified.")


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
    print("Testing Video FastAPI Endpoints...")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. Create a tiny synthetic 5-frame MP4
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
        # 2. Upload video
        with open(test_vid_path, "rb") as f:
            resp = client.post("/api/video/upload", files={"file": ("test_synthetic.mp4", f, "video/mp4")})
        assert resp.status_code == 200, f"Video upload failed: {resp.text}"
        upload_data = resp.json()
        assert upload_data["status"] == "success"
        assert upload_data["metadata"]["fps"] == 25.0
        assert upload_data["metadata"]["total_frames"] > 0
        print("  POST /api/video/upload verified (standardized to 25 FPS).")

        # 3. Process video
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

        # 4. Status polling (wait for completion)
        import time
        completed = False
        for _ in range(120):
            time.sleep(0.4)
            s_resp = client.get(f"/api/video/status/{job_id}")
            assert s_resp.status_code == 200
            s_data = s_resp.json()
            if s_data["state"] in ("completed", "error"):
                completed = True
                break

        assert completed is True, f"Job did not finish in time: {s_data}"
        assert s_data["state"] == "completed", f"Job failed: {s_data.get('error')}"
        assert s_data["current_frame"] == s_data["total_frames"]
        print(f"  GET /api/video/status verified: completed {s_data['current_frame']} frames at 25 FPS.")

        # 5. Frame annotations
        f_resp = client.get(f"/api/video/frame/{job_id}/0")
        assert f_resp.status_code == 200
        f_data = f_resp.json()
        assert "tracks" in f_data
        print(f"  GET /api/video/frame/0 verified: {f_data['active_count']} active tracks.")

        # 6. Export CSV
        exp_resp = client.get(f"/api/video/export/{job_id}?format=csv")
        assert exp_resp.status_code == 200
        assert len(exp_resp.content) > 0
        print("  GET /api/video/export (CSV) verified.")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_real_video_sample_pipeline():
    """Verifies end-to-end ingestion, tracking, keyframing, and export on real underwater reef footage."""
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
    assert meta["width"] == 972 and meta["height"] == 720
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
    import time
    for poll_idx in range(600):
        time.sleep(1.0)
        s_resp = client.get(f"/api/video/status/{job_id}")
        assert s_resp.status_code == 200
        s_data = s_resp.json()
        if poll_idx % 15 == 0:
            print(f"    Progress: frame {s_data.get('current_frame', 0)}/{s_data.get('total_frames', 0)} ({s_data.get('progress_pct', 0.0):.1f}%) | state={s_data.get('state')}")
        if s_data["state"] in ("completed", "error"):
            break

    assert s_data["state"] == "completed", f"Job failed: {s_data.get('error')}"
    print(f"  Completed! Processed {s_data['current_frame']} frames | Keyframes: {s_data['keyframes_count']} | Total corals: {s_data['total_unique_corals']}")
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
    test_keyframe_selector()
    test_benthic_tracker()
    test_export_utilities()
    test_video_api_endpoints()
    test_real_video_sample_pipeline()
    print("\nAll video pipeline unit tests PASSED successfully!")
