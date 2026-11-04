"""
Camera Ego-Motion Estimation and Adaptive Keyframe Selection for Underwater Marine Video.
Estimates camera translation and seafloor overlap ratio to trigger heavy model inference only
when significant new benthic area enters the camera's field of view.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class EgoMotionKeyframeSelector:
    """
    Selects keyframes based on cumulative camera ego-motion and benthic scene overlap.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.75,
        min_stride: int = 5,
        max_stride: int = 30,
        scene_cut_threshold: float = 0.45,
    ):
        """
        Args:
            overlap_threshold: Ratio of scene overlap remaining with anchor keyframe.
                               If overlap drops below this (e.g. 0.75 = 25% new seabed), a keyframe triggers.
            min_stride: Minimum frames between consecutive keyframes (prevents thrashing on camera shake).
            max_stride: Maximum frames between keyframes (safety temporal refresh, e.g. 1.0s at 30fps).
            scene_cut_threshold: Histogram correlation drop threshold indicating an abrupt cut or turbidity spike.
        """
        self.overlap_threshold = float(overlap_threshold)
        self.min_stride = max(1, int(min_stride))
        self.max_stride = max(self.min_stride, int(max_stride))
        self.scene_cut_threshold = float(scene_cut_threshold)

        self._keyframe_gray: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_motion_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._cumulative_H: np.ndarray = np.eye(3, dtype=np.float32)
        self._frames_since_keyframe = 0
        self._keyframe_count = 0
        self._total_frames = 0

    def reset(self):
        """Resets the selector state for a new video stream."""
        self._keyframe_gray = None
        self._prev_gray = None
        self._prev_motion_gray = None
        self._prev_hist = None
        self._cumulative_H = np.eye(3, dtype=np.float32)
        self._frames_since_keyframe = 0
        self._keyframe_count = 0
        self._total_frames = 0

    def evaluate_frame(
        self,
        frame_rgb: np.ndarray,
        force_keyframe: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluates an incoming RGB video frame and determines whether it qualifies as a Keyframe.

        Returns:
            (is_keyframe: bool, metrics: Dict[str, Any])
        """
        self._total_frames += 1
        h, w = frame_rgb.shape[:2]

        if not _HAS_CV2:
            # Fallback when OpenCV is not present: periodic stride
            is_kf = (self._total_frames == 1) or (self._frames_since_keyframe >= self.max_stride) or force_keyframe
            if is_kf:
                self._frames_since_keyframe = 0
                self._keyframe_count += 1
            else:
                self._frames_since_keyframe += 1
            return is_kf, {
                "overlap_ratio": 1.0 if not is_kf else 0.0,
                "motion_dx": 0.0,
                "motion_dy": 0.0,
                "is_scene_cut": False,
                "frames_since_kf": self._frames_since_keyframe,
                "keyframe_index": self._keyframe_count,
            }

        # Convert to grayscale for motion estimation
        curr_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Downscale proxy for optical flow (max_dim=640) to accelerate feature tracking 5-10x
        proxy_scale = 1.0
        if max(h, w) > 640:
            proxy_scale = 640.0 / float(max(h, w))
            proxy_w = max(16, int(round(w * proxy_scale)))
            proxy_h = max(16, int(round(h * proxy_scale)))
            curr_motion_gray = cv2.resize(curr_gray, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
        else:
            curr_motion_gray = curr_gray

        # 1. First frame is always a Keyframe
        if self._keyframe_gray is None or self._total_frames == 1 or force_keyframe:
            self._keyframe_gray = curr_gray.copy()
            self._prev_gray = curr_gray.copy()
            self._prev_motion_gray = curr_motion_gray.copy()
            self._prev_hist = self._compute_hist(curr_gray)
            self._cumulative_H = np.eye(3, dtype=np.float32)
            self._frames_since_keyframe = 0
            self._keyframe_count += 1
            return True, {
                "overlap_ratio": 1.0,
                "motion_dx": 0.0,
                "motion_dy": 0.0,
                "is_scene_cut": False,
                "frames_since_kf": 0,
                "keyframe_index": self._keyframe_count,
                "trigger_reason": "initial_frame" if not force_keyframe else "forced",
                "step_H": np.eye(3, dtype=np.float32),
                "cumulative_H": self._cumulative_H.copy(),
            }

        self._frames_since_keyframe += 1

        # 2. Check scene cut / turbidity burst via histogram correlation
        curr_hist = self._compute_hist(curr_gray)
        is_scene_cut = False
        if self._prev_hist is not None:
            hist_sim = cv2.compareHist(self._prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            if hist_sim < (1.0 - self.scene_cut_threshold):
                is_scene_cut = True
        self._prev_hist = curr_hist

        # 3. Inter-frame motion estimation using pyramidal Lucas-Kanade on proxy resolution
        pts_prev = cv2.goodFeaturesToTrack(
            self._prev_motion_gray if self._prev_motion_gray is not None else curr_motion_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

        step_H = np.eye(3, dtype=np.float32)
        dx, dy = 0.0, 0.0

        if pts_prev is not None and len(pts_prev) >= 8 and self._prev_motion_gray is not None:
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_motion_gray,
                curr_motion_gray,
                pts_prev,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            good_prev = pts_prev[status == 1]
            good_curr = pts_curr[status == 1]

            if len(good_prev) >= 6:
                # Estimate robust affine transform (translation + rotation + scale) with RANSAC
                affine, inliers = cv2.estimateAffinePartial2D(
                    good_prev,
                    good_curr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                )
                if affine is not None:
                    # Rescale translation offsets back to full-resolution coordinates
                    affine[0, 2] /= proxy_scale
                    affine[1, 2] /= proxy_scale
                    step_H = np.eye(3, dtype=np.float32)
                    step_H[:2, :] = affine
                    dx = float(affine[0, 2])
                    dy = float(affine[1, 2])

        self._prev_gray = curr_gray.copy()
        self._prev_motion_gray = curr_motion_gray.copy()

        # Accumulate homography relative to current keyframe anchor
        self._cumulative_H = self._cumulative_H @ step_H

        # 4. Calculate benthic overlap ratio of current frame with keyframe
        overlap_ratio = self._calculate_scene_overlap(w, h, self._cumulative_H)

        # 5. Evaluate Keyframe Trigger Decisions
        trigger_reason = None
        is_keyframe = False

        if is_scene_cut and self._frames_since_keyframe >= self.min_stride:
            is_keyframe = True
            trigger_reason = "scene_cut"
        elif overlap_ratio < self.overlap_threshold and self._frames_since_keyframe >= self.min_stride:
            is_keyframe = True
            trigger_reason = "ego_motion_displacement"
        elif self._frames_since_keyframe >= self.max_stride:
            is_keyframe = True
            trigger_reason = "temporal_cadence_timeout"

        metrics = {
            "overlap_ratio": round(overlap_ratio, 4),
            "motion_dx": round(dx, 2),
            "motion_dy": round(dy, 2),
            "is_scene_cut": is_scene_cut,
            "frames_since_kf": self._frames_since_keyframe,
            "keyframe_index": self._keyframe_count + (1 if is_keyframe else 0),
            "trigger_reason": trigger_reason or "tracking_frame",
            "step_H": step_H,
            "cumulative_H": self._cumulative_H.copy(),
        }

        if is_keyframe:
            # Reset anchor to current frame
            self._keyframe_gray = curr_gray.copy()
            self._cumulative_H = np.eye(3, dtype=np.float32)
            self._frames_since_keyframe = 0
            self._keyframe_count += 1

        return is_keyframe, metrics

    def _compute_hist(self, gray: np.ndarray) -> np.ndarray:
        """Computes normalized 1D histogram of grayscale pixel intensities."""
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _calculate_scene_overlap(self, w: int, h: int, H: np.ndarray) -> float:
        """
        Warps frame corner vertices using cumulative homography H and calculates
        the true polygon intersection area with the original viewport.
        """
        if not _HAS_CV2:
            return 1.0

        p1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        try:
            # Transform p1 by H to get the new warped quadrilateral in current coordinates
            p1_hom = np.hstack([p1, np.ones((4, 1), dtype=np.float32)])
            p2_hom = (H @ p1_hom.T).T
            p2 = p2_hom[:, :2] / np.maximum(p2_hom[:, 2:], 1e-6)

            # Compute exact polygon intersection area between viewport and warped quad
            inter_area, _ = cv2.intersectConvexConvex(p1, p2.astype(np.float32))
            total_area = float(w * h)
            overlap = float(inter_area) / max(total_area, 1.0)
            return float(np.clip(overlap, 0.0, 1.0))
        except Exception:
            return 1.0
