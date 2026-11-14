"""
Video Overlay Rendering, MP4 Compilation, Transect CSV Generation, and MOT/COCO-Video Serialization.
Utilizes fast vectorized OpenCV drawing operations for real-time video playback and export.
"""

import os
import json
import csv
import shutil
import subprocess
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from core.video.tracker import CoralTrack


def render_video_overlay(
    frame_bgr: np.ndarray,
    active_tracks: List[CoralTrack],
    draw_polygons: bool = True,
    draw_boxes: bool = False,
    draw_labels: bool = True,
    draw_hud: bool = True,
    frame_idx: int = 0,
    total_frames: int = 0,
    fps: float = 30.0,
    bleaching_prevalence_pct: float = 0.0,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Renders high-contrast segmentation masks, bounding boxes, species tags, and benthic HUD
    directly onto an OpenCV BGR video frame in ~2-4ms.
    """
    if not _HAS_CV2 or len(active_tracks) == 0 and not draw_hud:
        return frame_bgr

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # 1. Render Polygons with Alpha Blending
    if draw_polygons:
        overlay = out.copy()
        for track in active_tracks:
            poly = getattr(track, "polygon", None) or (track.get("polygon") if isinstance(track, dict) else None)
            if not poly or len(poly) < 3:
                continue

            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            color_rgb = getattr(track, "color_rgb", None) or (track.get("color_rgb") if isinstance(track, dict) else None) or [46, 204, 113]
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.polylines(out, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    # 2. Render Bounding Boxes & Labels
    for track in active_tracks:
        bbox = getattr(track, "bbox", None) or (track.get("bbox") if isinstance(track, dict) else None) or [0, 0, 0, 0]
        x, y, bw, bh = bbox
        color_rgb = getattr(track, "color_rgb", None) or (track.get("color_rgb") if isinstance(track, dict) else None) or [46, 204, 113]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        if draw_boxes:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color_bgr, thickness=2, lineType=cv2.LINE_AA)

        if draw_labels:
            taxonomy = getattr(track, "taxonomy", None) or (track.get("taxonomy") if isinstance(track, dict) else None) or {}
            bleaching = getattr(track, "bleaching", None) or (track.get("bleaching") if isinstance(track, dict) else None) or {}
            track_id = getattr(track, "track_id", None) or (track.get("track_id") if isinstance(track, dict) else None) or 0

            genus = (taxonomy.get("genus") if taxonomy else None) or (track.get("genus") if isinstance(track, dict) else None) or "Coral"
            is_bleached = (bleaching.get("is_bleached") if bleaching else None) or (track.get("is_bleached", False) if isinstance(track, dict) else False)
            cond_str = (track.get("condition") if isinstance(track, dict) else None) or ("Bleached" if is_bleached else "Healthy")
            label = f"#{track_id} {genus} [{cond_str}]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_x = max(0, x)
            badge_y = max(text_h + 4, y - 6)

            # Draw background pill
            badge_bg = (20, 20, 20)
            cv2.rectangle(
                out,
                (badge_x, badge_y - text_h - 4),
                (badge_x + text_w + 6, badge_y + 2),
                badge_bg,
                -1,
            )
            # Text colored red if bleached, else track color
            text_color = (68, 68, 239) if is_bleached else (240, 240, 240)
            cv2.putText(
                out,
                label,
                (badge_x + 3, badge_y - 2),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA,
            )

    # 3. Render Top Benthic HUD
    if draw_hud:
        hud_h = 36
        hud_overlay = out[:hud_h, :].copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (12, 16, 24), -1)
        cv2.addWeighted(hud_overlay, 0.85, out[:hud_h, :], 0.15, 0, out[:hud_h, :])

        sec = frame_idx / max(fps, 1.0)
        hud_text = (
            f"ReefVision Transect Monitor | Frame {frame_idx + 1}/{total_frames} ({sec:.1f}s) | "
            f"Active Corals: {len(active_tracks)} | "
            f"Bleaching: {bleaching_prevalence_pct:.1f}% | "
            f"{fps:.1f} FPS"
        )
        cv2.putText(
            out,
            hud_text,
            (14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 240, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return out


def _test_encoder_works(ffmpeg_bin: str, encoder: str) -> bool:
    """Verifies that an encoder actually initializes and executes without driver/hardware error."""
    try:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", "color=c=black:s=64x64:d=0.04",
            "-c:v", encoder,
            "-f", "null",
            "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=4)
        return res.returncode == 0
    except Exception:
        return False


def _detect_fastest_h264_encoder(ffmpeg_bin: str) -> str:
    """Detects available and hardware-validated encoder: VideoToolbox (Mac), NVENC (CUDA), or ultrafast libx264."""
    try:
        import platform
        res = subprocess.run([ffmpeg_bin, "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3)
        stdout = res.stdout or ""
        if platform.system() == "Darwin" and "h264_videotoolbox" in stdout:
            if _test_encoder_works(ffmpeg_bin, "h264_videotoolbox"):
                return "h264_videotoolbox"
        if "h264_nvenc" in stdout:
            if _test_encoder_works(ffmpeg_bin, "h264_nvenc"):
                return "h264_nvenc"
    except Exception:
        pass
    if _test_encoder_works(ffmpeg_bin, "libx264"):
        return "libx264"
    return "libx264"


def export_annotated_video(
    input_video_path: str,
    output_mp4_path: str,
    frame_tracks_map: Dict[int, List[CoralTrack]],
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    max_frames: Optional[int] = None,
) -> str:
    """
    Generates an annotated MP4 video with rendered bounding boxes, polygons, and benthic HUD.
    Uses single-pass hardware-accelerated ffmpeg streaming pipe with verified encoder initialization,
    captures actionable errors, validates output dimensions and frame count, and falls back to OpenCV VideoWriter.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required to export video.")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {input_video_path}")

    raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(raw_total_frames, max_frames) if max_frames is not None and max_frames > 0 else raw_total_frames
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    ffmpeg_proc = None
    writer = None

    if ffmpeg_bin:
        encoder = _detect_fastest_h264_encoder(ffmpeg_bin)
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", encoder,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if encoder == "libx264":
            ffmpeg_cmd.extend(["-preset", "ultrafast", "-crf", "22"])
        ffmpeg_cmd.append(output_mp4_path)
        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception:
            ffmpeg_proc = None

    if ffmpeg_proc is None:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

    ffmpeg_error_msg = None
    try:
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            tracks = frame_tracks_map.get(frame_idx, [])
            bleached_count = 0
            for t in tracks:
                b_dict = getattr(t, "bleaching", None) or (t.get("bleaching") if isinstance(t, dict) else {}) or {}
                if b_dict.get("is_bleached", False):
                    bleached_count += 1
            bleach_pct = (bleached_count / max(len(tracks), 1)) * 100.0

            rendered = render_video_overlay(
                frame_bgr=frame_bgr,
                active_tracks=tracks,
                draw_polygons=True,
                draw_boxes=False,
                draw_labels=True,
                draw_hud=True,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
                bleaching_prevalence_pct=bleach_pct,
            )
            if ffmpeg_proc and ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.write(rendered.tobytes())
                except BrokenPipeError:
                    _, err_bytes = ffmpeg_proc.communicate()
                    ffmpeg_error_msg = err_bytes.decode("utf-8", errors="replace")[-500:] if err_bytes else "Broken pipe"
                    break
            elif writer:
                writer.write(rendered)

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx, total_frames, (frame_idx / total_frames) * 100.0)

    finally:
        cap.release()
        if ffmpeg_proc:
            try:
                err_bytes = None
                if ffmpeg_proc.poll() is None:
                    # communicate() closes stdin itself and waits for the muxer to finalize the file. Closing stdin
                    # first makes communicate() raise "flush of closed file" and skips the wait, leaving an
                    # unfinished MP4 (no moov atom) that fails verification.
                    _, err_bytes = ffmpeg_proc.communicate(timeout=120)
                if ffmpeg_proc.returncode != 0 and not ffmpeg_error_msg:
                    ffmpeg_error_msg = err_bytes.decode("utf-8", errors="replace")[-500:] if err_bytes else f"Exit code {ffmpeg_proc.returncode}"
            except Exception as e:
                if not ffmpeg_error_msg:
                    ffmpeg_error_msg = str(e)
        if writer:
            writer.release()

            if ffmpeg_bin and os.path.isfile(output_mp4_path) and os.path.getsize(output_mp4_path) > 0:
                temp_faststart = output_mp4_path + ".faststart.mp4"
                try:
                    subprocess.run(
                        [
                            ffmpeg_bin,
                            "-y",
                            "-i", output_mp4_path,
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-preset", "ultrafast",
                            "-movflags", "+faststart",
                            temp_faststart,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                    if os.path.isfile(temp_faststart) and os.path.getsize(temp_faststart) > 0:
                        os.replace(temp_faststart, output_mp4_path)
                except Exception:
                    if os.path.isfile(temp_faststart):
                        try:
                            os.remove(temp_faststart)
                        except Exception:
                            pass

    if ffmpeg_error_msg and (not os.path.isfile(output_mp4_path) or os.path.getsize(output_mp4_path) == 0):
        raise RuntimeError(f"FFmpeg single-pass export failed: {ffmpeg_error_msg}")

    # Validate generated output file
    if not os.path.isfile(output_mp4_path) or os.path.getsize(output_mp4_path) == 0:
        raise RuntimeError(f"Export output MP4 is missing or empty: {output_mp4_path}")

    cap_verify = cv2.VideoCapture(output_mp4_path)
    if not cap_verify.isOpened():
        raise RuntimeError(f"Exported MP4 is unreadable or corrupted: {output_mp4_path}")
    v_w = int(cap_verify.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap_verify.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_frames = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_verify.release()

    if v_w != width or v_h != height:
        raise ValueError(f"Exported video dimensions ({v_w}x{v_h}) do not match source ({width}x{height})")
    if total_frames > 0 and v_frames < max(1, total_frames - 5):
        raise ValueError(f"Exported frame count ({v_frames}) significantly below expected ({total_frames})")

    return output_mp4_path


def export_transect_csv(
    output_csv_path: str,
    time_series_records: List[Dict[str, Any]],
) -> str:
    """
    Exports frame-by-frame scientific transect statistics to CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "active_corals_count",
        "cumulative_unique_corals",
        "bleached_corals_count",
        "healthy_corals_count",
        "bleaching_prevalence_pct",
        "dominant_genus",
        "is_keyframe",
        "camera_motion_dx",
        "camera_motion_dy",
    ]

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in time_series_records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})

    return output_csv_path


def export_coco_video(
    output_json_path: str,
    video_filename: str,
    width: int,
    height: int,
    total_frames: int,
    fps: float,
    all_tracks: List[CoralTrack],
    frame_tracks_map: Dict[int, List[CoralTrack]],
) -> str:
    """
    Exports tracking annotations to MOT / COCO-Video format JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)

    videos = [{
        "id": 1,
        "name": video_filename,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
    }]

    categories = [
        {"id": 1, "name": "Coral Instance", "supercategory": "Benthos"},
    ]

    annotations = []
    ann_id = 1
    for frame_idx, tracks in frame_tracks_map.items():
        for t in tracks:
            tid = getattr(t, "track_id", None) or (t.get("track_id") if isinstance(t, dict) else 0)
            bbox = getattr(t, "bbox", None) or (t.get("bbox") if isinstance(t, dict) else [0, 0, 0, 0])
            area_px = getattr(t, "area_px", None) or (t.get("area_px") if isinstance(t, dict) else 0)
            pred_iou = getattr(t, "predicted_iou", None) if hasattr(t, "predicted_iou") else (t.get("predicted_iou") if isinstance(t, dict) else None)
            tax = getattr(t, "taxonomy", None) or (t.get("taxonomy") if isinstance(t, dict) else {}) or {}
            bl = getattr(t, "bleaching", None) or (t.get("bleaching") if isinstance(t, dict) else {}) or {}

            annotations.append({
                "id": ann_id,
                "video_id": 1,
                "frame_id": frame_idx + 1,
                "track_id": tid,
                "category_id": 1,
                "bbox": bbox,
                "area": area_px,
                "predicted_iou": pred_iou,
                "taxonomy": (tax.get("genus") if tax else None) or (t.get("genus") if isinstance(t, dict) else None) or "Coral",
                "growth_form": (tax.get("growth_form") if tax else None) or (t.get("growth_form") if isinstance(t, dict) else None) or "-",
                "condition": (bl.get("status") if bl else None) or (t.get("condition") if isinstance(t, dict) else None) or "Healthy",
                "is_bleached": (bl.get("is_bleached") if bl else None) or (t.get("is_bleached", False) if isinstance(t, dict) else False),
            })
            ann_id += 1

    payload = {
        "info": {
            "description": "Andromeida ReefVision Video Transect Tracking",
            "version": "2.0",
        },
        "videos": videos,
        "categories": categories,
        "annotations": annotations,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_json_path
