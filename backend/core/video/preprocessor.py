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


def prepare_bounded_chunk_frames(
    video_path: str,
    chunk_dir: str,
    start_frame: int,
    end_frame: int,
    max_dim: int = 480,
    jpeg_quality: int = 80,
    keyframe_selector: Optional[Any] = None,
    min_motion_frame: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Decodes video sequentially for half-open range [start_frame, end_frame).
    Saves downscaled proxy JPEGs (00000.jpg, 00001.jpg, ...) into chunk_dir.
    Evaluates ego-motion keyframe selection strictly in chronological order for frames >= min_motion_frame,
    preventing duplicate evaluation of overlap frames across chunk boundaries.
    Validates frame counts, writes, and dimensions.
    Returns: (list_of_jpeg_paths, metadata_dict)
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if start_frame < 0:
        raise ValueError(f"start_frame cannot be negative: {start_frame}")
    if end_frame <= start_frame:
        raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame})")

    os.makedirs(chunk_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if orig_w <= 0 or orig_h <= 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions {orig_w}x{orig_h} for: {video_path}")

    scale = 1.0
    if orig_w > max_dim or orig_h > max_dim:
        scale = min(max_dim / orig_w, max_dim / orig_h)

    proxy_w = max(16, int(round(orig_w * scale)))
    proxy_h = max(16, int(round(orig_h * scale)))
    scale_x = orig_w / proxy_w if proxy_w > 0 else 1.0
    scale_y = orig_h / proxy_h if proxy_h > 0 else 1.0

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_paths = []
    chunk_keyframes = []
    chunk_motion_metas = {}
    expected_frames = end_frame - start_frame

    for rel_idx in range(expected_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if scale < 0.999:
            frame = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)

        abs_f = start_frame + rel_idx
        if keyframe_selector is not None and abs_f >= min_motion_frame:
            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            is_kf, m_meta = keyframe_selector.evaluate_frame(f_rgb)
            chunk_motion_metas[abs_f] = m_meta
            if is_kf and abs_f > 0:
                chunk_keyframes.append(abs_f)

        # SAM2 requires 00000.jpg, 00001.jpg, ... within the directory
        frame_name = f"{rel_idx:05d}.jpg"
        frame_path = os.path.join(chunk_dir, frame_name)

        write_ok = cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not write_ok or not os.path.isfile(frame_path) or os.path.getsize(frame_path) == 0:
            cap.release()
            raise RuntimeError(f"Failed to write chunk frame image: {frame_path}")

        frame_paths.append(frame_path)

    cap.release()

    meta = {
        "start_frame": start_frame,
        "end_frame": start_frame + len(frame_paths),
        "chunk_frames_count": len(frame_paths),
        "total_video_frames": total_video_frames,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "proxy_w": proxy_w,
        "proxy_h": proxy_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "chunk_keyframes": chunk_keyframes,
        "chunk_motion_metas": chunk_motion_metas,
    }
    return frame_paths, meta


def cleanup_chunk_dir(chunk_dir: str):
    """Safely removes a temporary chunk frames directory and its contents."""
    if chunk_dir and os.path.isdir(chunk_dir):
        shutil.rmtree(chunk_dir, ignore_errors=True)


def extract_video_frames_to_disk(
    video_path: str,
    frames_dir: str,
    max_dim: int = 480,
    jpeg_quality: int = 80,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extracts video frames to dual-resolution JPEG images in frames_dir (00000.jpg, 00001.jpg, ...)
    for SAM 2 video predictor initialization.
    Supports optional bounding via [start_frame, end_frame).
    """
    if end_frame is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        end_frame = max(1, total)

    return prepare_bounded_chunk_frames(
        video_path=video_path,
        chunk_dir=frames_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        max_dim=max_dim,
        jpeg_quality=jpeg_quality,
    )


class BoundedVideoFrameBuffer:
    """
    Sequential, memory-bounded in-memory frame buffer for temporal chunk execution.
    - Decodes video sequentially forward without redundant seeks or re-reading.
    - Retains overlap frames from the tail of the previous chunk so they are NOT re-decoded.
    - Maintains proxy RGB numpy arrays for SAM/TAM tracking and motion estimation.
    - Retains full-resolution keyframe images for BioCLIP/YOLO crops and CoralSCOP discovery.
    - Completely eliminates temporary JPEG disk writing, encoding, and reading round-trips.
    """

    def __init__(
        self,
        video_path: str,
        max_dim: int = 480,
    ):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.max_dim = max_dim

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0

        if self.orig_w <= 0 or self.orig_h <= 0:
            self.cap.release()
            raise ValueError(f"Invalid video dimensions {self.orig_w}x{self.orig_h} for: {video_path}")

        scale = 1.0
        if self.orig_w > max_dim or self.orig_h > max_dim:
            scale = min(max_dim / self.orig_w, max_dim / self.orig_h)

        self.proxy_w = max(16, int(round(self.orig_w * scale)))
        self.proxy_h = max(16, int(round(self.orig_h * scale)))
        self.scale_x = self.orig_w / self.proxy_w if self.proxy_w > 0 else 1.0
        self.scale_y = self.orig_h / self.proxy_h if self.proxy_h > 0 else 1.0
        self.scale = scale

        self.cur_decoded_idx = 0
        self.proxy_frames: Dict[int, np.ndarray] = {}
        self.raw_kf_cache: Dict[int, np.ndarray] = {}

    def fetch_chunk(
        self,
        start_frame: int,
        end_frame: int,
        keyframe_selector: Optional[Any] = None,
        min_motion_frame: int = 0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Retrieves or decodes proxy frames for [start_frame, end_frame).
        Retains overlap frames from previous chunk without re-decoding.
        Evaluates motion estimation chronologically for frames >= min_motion_frame.
        """
        # 1. Prune frames older than start_frame to bound memory footprint
        old_keys = [k for k in self.proxy_frames if k < start_frame]
        for k in old_keys:
            self.proxy_frames.pop(k, None)
        old_kf_keys = [k for k in self.raw_kf_cache if k < start_frame - 60]
        for k in old_kf_keys:
            self.raw_kf_cache.pop(k, None)

        chunk_keyframes = []
        chunk_motion_metas = {}

        # 2. If we need to seek backward, reposition cap
        if self.cur_decoded_idx > start_frame and start_frame not in self.proxy_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.cur_decoded_idx = start_frame

        # 3. Decode missing frames forward sequentially
        while self.cur_decoded_idx < end_frame:
            f_idx = self.cur_decoded_idx
            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                break

            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize to proxy resolution
            if self.scale < 0.999:
                proxy_rgb = cv2.resize(frame_rgb, (self.proxy_w, self.proxy_h), interpolation=cv2.INTER_AREA)
            else:
                proxy_rgb = frame_rgb

            self.proxy_frames[f_idx] = proxy_rgb

            # Chronological motion estimation
            if keyframe_selector is not None and f_idx >= min_motion_frame:
                is_kf, m_meta = keyframe_selector.evaluate_frame(proxy_rgb)
                chunk_motion_metas[f_idx] = m_meta
                if is_kf and f_idx > 0:
                    chunk_keyframes.append(f_idx)
                    self.raw_kf_cache[f_idx] = frame_rgb
            elif f_idx in self.raw_kf_cache:
                pass

            self.cur_decoded_idx += 1

        chunk_frames = [self.proxy_frames[i] for i in range(start_frame, min(end_frame, self.cur_decoded_idx)) if i in self.proxy_frames]

        meta = {
            "start_frame": start_frame,
            "end_frame": start_frame + len(chunk_frames),
            "chunk_frames_count": len(chunk_frames),
            "total_video_frames": self.total_frames,
            "orig_w": self.orig_w,
            "orig_h": self.orig_h,
            "proxy_w": self.proxy_w,
            "proxy_h": self.proxy_h,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "chunk_keyframes": chunk_keyframes,
            "chunk_motion_metas": chunk_motion_metas,
        }
        return chunk_frames, meta

    def get_keyframe_raw_rgb(self, frame_idx: int) -> np.ndarray:
        """Returns the full-resolution RGB image for keyframe crop extraction."""
        if frame_idx in self.raw_kf_cache:
            return self.raw_kf_cache[frame_idx]

        # Single seek fallback if not in cache
        cur_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
        if ret and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.raw_kf_cache[frame_idx] = rgb
            return rgb
        # Fallback to proxy frame if seek failed
        if frame_idx in self.proxy_frames:
            return self.proxy_frames[frame_idx]
        return np.zeros((self.orig_h, self.orig_w, 3), dtype=np.uint8)

    def close(self):
        """Releases video capture and clears frame buffers."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.proxy_frames.clear()
        self.raw_kf_cache.clear()
