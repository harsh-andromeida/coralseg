import os
import shutil
import subprocess
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def standardize_video_to_25fps(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Standardizes any video to a constant 25.0 FPS grid.
    Guarantees 1 second = exactly 25 frames (Frame 0 = 0.0s, Frame 25 = 1.0s, Frame 50 = 2.0s).
    Uses web-standard H.264 (avc1) with +faststart (moov atom at start) for instant browser playback.

    Guarantees atomic output replacement; never uses the same path as both input and output.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Video file not found: {input_path}")

    target_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(target_dir, exist_ok=True)

    # Use unique temporary output to prevent reading and writing the same file
    temp_output_path = os.path.join(target_dir, f".tmp_{os.getpid()}_{int(time.time() * 1000)}_{os.path.basename(output_path)}")

    ffmpeg_bin = shutil.which("ffmpeg")
    success = False

    # 1. Preferred method: Fast, universal H.264 browser-streamable conversion via ffmpeg
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", input_path,
                "-r", "25",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-movflags", "+faststart",
                temp_output_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.isfile(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                success = True
        except Exception as ffmpeg_err:
            print(f"[preprocessor] ffmpeg conversion failed, falling back to OpenCV: {ffmpeg_err}")

    # 2. Fallback: OpenCV frame-by-frame resampling
    if not success:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            raise ValueError(f"Could not open video file: {input_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if orig_fps <= 0 or orig_total <= 0:
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

        writer = None
        for codec_tag in ("avc1", "H264", "x264", "mp4v"):
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_tag)
                test_writer = cv2.VideoWriter(temp_output_path, fourcc, 25.0, (orig_w, orig_h))
                if test_writer.isOpened():
                    writer = test_writer
                    break
            except Exception:
                continue

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_output_path, fourcc, 25.0, (orig_w, orig_h))

        cur_src_idx = -1
        cur_frame = None

        try:
            for out_idx in range(target_total_frames):
                target_t = out_idx / 25.0
                target_src_idx = min(orig_total - 1, int(round(target_t * orig_fps)))

                while cur_src_idx < target_src_idx:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cur_src_idx += 1
                    cur_frame = frame

                if cur_frame is not None:
                    writer.write(cur_frame)
            success = True
        finally:
            cap.release()
            if writer:
                writer.release()

        # Move moov atom to beginning if ffmpeg is available
        if success and ffmpeg_bin and os.path.isfile(temp_output_path):
            temp_faststart = temp_output_path + ".faststart.mp4"
            try:
                subprocess.run(
                    [ffmpeg_bin, "-y", "-i", temp_output_path, "-c", "copy", "-movflags", "+faststart", temp_faststart],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                if os.path.isfile(temp_faststart) and os.path.getsize(temp_faststart) > 0:
                    os.replace(temp_faststart, temp_output_path)
            except Exception:
                if os.path.isfile(temp_faststart):
                    try:
                        os.remove(temp_faststart)
                    except Exception:
                        pass

    if not success or not os.path.isfile(temp_output_path) or os.path.getsize(temp_output_path) == 0:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise RuntimeError(f"Failed to transcode video to 25 FPS: {input_path}")

    # Validate the written temporary file
    cap_verify = cv2.VideoCapture(temp_output_path)
    if not cap_verify.isOpened():
        os.remove(temp_output_path)
        raise RuntimeError(f"Written video is corrupted or unreadable: {temp_output_path}")

    final_w = int(cap_verify.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_h = int(cap_verify.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final_total = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_verify.release()

    if final_total <= 0:
        os.remove(temp_output_path)
        raise RuntimeError(f"Transcoded video contains 0 frames: {temp_output_path}")

    # Atomic rename/replace onto the destination
    os.replace(temp_output_path, output_path)

    return {
        "width": final_w,
        "height": final_h,
        "fps": 25.0,
        "total_frames": final_total,
        "duration_sec": round(final_total / 25.0, 2),
        "filename": os.path.basename(output_path),
        "file_size": os.path.getsize(output_path),
    }
