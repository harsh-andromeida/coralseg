import os
import cv2
from typing import Dict, Any, List, Tuple


def standardize_video_to_25fps(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Standardizes any video to a constant 25.0 FPS grid.
    Guarantees 1 second = exactly 25 frames (Frame 0 = 0.0s, Frame 25 = 1.0s, Frame 50 = 2.0s).
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if orig_fps <= 0 or orig_total <= 0:
        # Fallback probe by reading frames
        orig_fps = 25.0 if orig_fps <= 0 else orig_fps
        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        orig_total = frame_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    duration_sec = orig_total / max(orig_fps, 1.0)
    target_total_frames = max(1, int(round(duration_sec * 25.0)))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 25.0, (orig_w, orig_h))

    # Fast O(N) sequential frame resampling
    cur_src_idx = -1
    cur_frame = None

    for out_idx in range(target_total_frames):
        target_t = out_idx / 25.0
        target_src_idx = min(orig_total - 1, int(round(target_t * orig_fps)))

        # Advance capture until reaching target_src_idx
        while cur_src_idx < target_src_idx:
            ret, frame = cap.read()
            if not ret:
                break
            cur_src_idx += 1
            cur_frame = frame

        if cur_frame is not None:
            writer.write(cur_frame)

    cap.release()
    writer.release()

    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    return {
        "width": orig_w,
        "height": orig_h,
        "fps": 25.0,
        "total_frames": target_total_frames,
        "duration_sec": round(target_total_frames / 25.0, 2),
        "filename": os.path.basename(output_path),
        "file_size": file_size,
    }


def extract_video_frames_to_disk(
    video_path: str,
    frames_dir: str,
    max_dim: int = 720,
    jpeg_quality: int = 85,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extracts video frames to dual-resolution JPEG images in frames_dir (00000.jpg, 00001.jpg, ...)
    for SAM 2 video predictor initialization.
    Downscales to inference proxy resolution (e.g. max_dim=720) for 5-10x faster inference,
    while returning affine scaling factors (scale_x, scale_y) to map back to raw display coordinates.
    """
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if orig_w > 0 and orig_h > 0 and (orig_w > max_dim or orig_h > max_dim):
        scale = min(max_dim / orig_w, max_dim / orig_h)

    proxy_w = max(16, int(round(orig_w * scale)))
    proxy_h = max(16, int(round(orig_h * scale)))

    scale_x = orig_w / proxy_w if proxy_w > 0 else 1.0
    scale_y = orig_h / proxy_h if proxy_h > 0 else 1.0

    frame_paths = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale < 0.999:
            frame = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
        frame_name = f"{idx:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)
        # Write optimized JPEG for fast I/O
        cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        frame_paths.append(frame_path)
        idx += 1

    cap.release()
    meta = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "proxy_w": proxy_w,
        "proxy_h": proxy_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    return frame_paths, meta
