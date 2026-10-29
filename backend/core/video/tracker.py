"""
Benthic Coral Colony Tracker with Persistent IDs, Homography Warping, and IoU Association.
Maintains persistent track IDs across underwater video frames while camera translates over seafloor.
Decouples heavy taxonomy and bleaching inference so they only run on newly entered colonies.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from core.visualization import generate_distinct_colors


class CoralTrack:
    """Represents a single tracked coral colony with persistent ID and attributes."""

    def __init__(
        self,
        track_id: int,
        bbox: List[int],
        polygon: List[List[float]],
        area_px: int,
        area_pct: float,
        predicted_iou: float,
        centroid: Tuple[int, int],
        color_rgb: Tuple[int, int, int],
        frame_idx: int,
        taxonomy: Optional[Dict[str, Any]] = None,
        bleaching: Optional[Dict[str, Any]] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.track_id = int(track_id)
        self.bbox = list(bbox)  # [x, y, w, h]
        self.polygon = list(polygon)  # [[x, y], ...]
        self.mask = mask
        self.area_px = int(area_px)
        self.area_pct = float(area_pct)
        self.predicted_iou = float(predicted_iou)
        self.centroid = tuple(centroid)
        self.color_rgb = tuple(color_rgb)
        self.color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
        self.first_seen_frame = int(frame_idx)
        self.last_seen_frame = int(frame_idx)

        # Default placeholder taxonomy and bleaching until evaluated
        self.taxonomy = taxonomy or {
            "genus": "Coral",
            "primary_label": f"Coral Colony #{track_id}",
            "growth_form": "Encrusting / Massive",
            "confidence": 0.85,
            "confidence_pct": 85.0,
            "color": self.color_rgb,
        }
        self.bleaching = bleaching or {
            "status": "Healthy",
            "is_bleached": False,
            "confidence": 0.90,
            "confidence_pct": 90.0,
            "bleached_prob": 0.10,
            "healthy_prob": 0.90,
        }
        self.needs_enrichment = (taxonomy is None or bleaching is None)

        self.hits = 1
        self.misses = 0
        self.active = True
        self.history = [(list(bbox), list(polygon))]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes track state for API responses and video overlay rendering."""
        return {
            "track_id": self.track_id,
            "id": self.track_id,
            "id_str": f"#{self.track_id}",
            "bbox": self.bbox,
            "polygon": self.polygon,
            "centroid": self.centroid,
            "area_px": self.area_px,
            "area_pct": self.area_pct,
            "predicted_iou": self.predicted_iou,
            "genus": self.taxonomy.get("genus", "Coral"),
            "primary_label": self.taxonomy.get("primary_label", f"Coral #{self.track_id}"),
            "growth_form": self.taxonomy.get("growth_form", "-"),
            "taxon_conf": self.taxonomy.get("confidence_pct", 85.0),
            "condition": self.bleaching.get("status", "Healthy"),
            "condition_conf": self.bleaching.get("confidence_pct", 90.0),
            "is_bleached": self.bleaching.get("is_bleached", False),
            "bleached_prob": self.bleaching.get("bleached_prob", 0.10),
            "color_rgb": self.color_rgb,
            "color_hex": self.color_hex,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "hits": self.hits,
            "active": self.active,
        }


def compute_box_iou(box1: List[int], box2: List[int]) -> float:
    """Computes Intersection-over-Union between two [x, y, w, h] bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return float(inter_area / max(union, 1))


class BenthicColonyTracker:
    """
    Tracks coral colonies across video frames using camera ego-motion homography warping
    and Hungarian / greedy IoU assignment on keyframes.
    """

    def __init__(
        self,
        iou_match_threshold: float = 0.30,
        max_lost_frames: int = 15,
        min_hits_to_confirm: int = 1,
    ):
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_lost_frames = int(max_lost_frames)
        self.min_hits_to_confirm = int(min_hits_to_confirm)

        self._next_track_id = 1
        self._tracks: Dict[int, CoralTrack] = {}
        self._palette = generate_distinct_colors(100)

    def reset(self):
        """Resets all active and historical tracks."""
        self._next_track_id = 1
        self._tracks.clear()

    def get_active_tracks(self) -> List[CoralTrack]:
        """Returns all currently active tracks within viewport."""
        return [t for t in self._tracks.values() if t.active]

    def get_tracks_needing_enrichment(self) -> List[CoralTrack]:
        """Returns tracks that have not yet undergone BioCLIP / YOLO Bleaching."""
        return [t for t in self._tracks.values() if t.active and t.needs_enrichment]

    def warp_tracks_with_motion(
        self,
        step_H: np.ndarray,
        frame_w: int,
        frame_h: int,
        frame_idx: int,
    ):
        """
        Warp all active coral polygons and bounding boxes forward using camera motion matrix step_H.
        Runs at >150 FPS on non-keyframes.
        """
        if not _HAS_CV2:
            return

        for track in list(self._tracks.values()):
            if not track.active:
                continue

            # 1. Warp polygon vertices
            if track.polygon and len(track.polygon) >= 3:
                pts = np.array(track.polygon, dtype=np.float32).reshape(-1, 1, 2)
                # Apply affine transformation (top 2x3 of step_H)
                warped_pts = cv2.transform(pts, step_H[:2, :])
                warped_poly = [[float(p[0][0]), float(p[0][1])] for p in warped_pts]

                # Compute new bounding box from warped polygon
                xs = [p[0] for p in warped_poly]
                ys = [p[1] for p in warped_poly]
                min_x = max(0, int(min(xs)))
                min_y = max(0, int(min(ys)))
                max_x = min(frame_w, int(max(xs)))
                max_y = min(frame_h, int(max(ys)))
                new_w = max_x - min_x
                new_h = max_y - min_y

                # If track completely moved out of frame, mark inactive
                if new_w < 5 or new_h < 5:
                    track.active = False
                    continue

                track.polygon = warped_poly
                track.bbox = [min_x, min_y, new_w, new_h]
                track.centroid = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
                track.last_seen_frame = frame_idx
                track.hits += 1
            else:
                # Warp bbox corners
                x, y, w, h = track.bbox
                corners = np.array([
                    [[x, y]],
                    [[x + w, y + h]],
                ], dtype=np.float32)
                warped = cv2.transform(corners, step_H[:2, :])
                x0, y0 = int(warped[0, 0, 0]), int(warped[0, 0, 1])
                x1, y1 = int(warped[1, 0, 0]), int(warped[1, 0, 1])
                x0 = max(0, min(frame_w, x0))
                y0 = max(0, min(frame_h, y0))
                x1 = max(0, min(frame_w, x1))
                y1 = max(0, min(frame_h, y1))
                bw = x1 - x0
                bh = y1 - y0

                if bw < 5 or bh < 5:
                    track.active = False
                    continue

                track.bbox = [x0, y0, bw, bh]
                track.centroid = (x0 + bw // 2, y0 + bh // 2)
                track.last_seen_frame = frame_idx
                track.hits += 1

    def match_keyframe_detections(
        self,
        new_detections: List[Dict[str, Any]],
        frame_idx: int,
        frame_w: int,
        frame_h: int,
    ) -> List[CoralTrack]:
        """
        Associates freshly segmented SAM masks on a Keyframe with existing active tracks.
        Updates matched tracks with refined boundaries; spawns new tracks for newly uncovered corals.
        """
        active_tracks = [t for t in self._tracks.values() if t.active]
        n_tracks = len(active_tracks)
        n_dets = len(new_detections)

        if n_tracks == 0:
            # All detections are newly spawned tracks
            for det in new_detections:
                self._spawn_track(det, frame_idx)
            return self.get_active_tracks()

        if n_dets == 0:
            # No detections on this keyframe; increment misses
            for track in active_tracks:
                track.misses += 1
                if track.misses > self.max_lost_frames:
                    track.active = False
            return self.get_active_tracks()

        # Compute IoU cost matrix
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, track in enumerate(active_tracks):
            for j, det in enumerate(new_detections):
                det_bbox = det.get("bbox", [0, 0, 1, 1])
                iou_matrix[i, j] = compute_box_iou(track.bbox, det_bbox)

        # Greedy bipartite matching
        matched_tracks = set()
        matched_dets = set()

        # Sort candidate pairs by descending IoU
        pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if iou_matrix[i, j] >= self.iou_match_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for iou_val, trk_idx, det_idx in pairs:
            if trk_idx in matched_tracks or det_idx in matched_dets:
                continue

            matched_tracks.add(trk_idx)
            matched_dets.add(det_idx)

            track = active_tracks[trk_idx]
            det = new_detections[det_idx]

            # Update geometry from fresh keyframe segmentation
            track.bbox = det.get("bbox", track.bbox)
            track.polygon = det.get("polygon", track.polygon)
            track.mask = det.get("segmentation", det.get("mask", track.mask))
            track.area_px = det.get("area_px", track.area_px)
            track.area_pct = det.get("area_pct", track.area_pct)
            track.predicted_iou = det.get("predicted_iou", track.predicted_iou)
            track.centroid = det.get("centroid", track.centroid)
            track.last_seen_frame = frame_idx
            track.misses = 0
            track.hits += 1

            # Retain established taxonomy and bleaching unless explicitly provided
            if "taxonomy" in det and det["taxonomy"]:
                track.taxonomy = det["taxonomy"]
            if "bleaching" in det and det["bleaching"]:
                track.bleaching = det["bleaching"]

        # Handle unmatched active tracks
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.misses += 1
                if track.misses > self.max_lost_frames:
                    track.active = False

        # Spawn new tracks for unmatched new detections
        for j, det in enumerate(new_detections):
            if j not in matched_dets:
                self._spawn_track(det, frame_idx)

        return self.get_active_tracks()

    def _spawn_track(self, det: Dict[str, Any], frame_idx: int) -> CoralTrack:
        """Spawns a new persistent CoralTrack from a detection dictionary."""
        track_id = self._next_track_id
        self._next_track_id += 1

        color_idx = (track_id - 1) % len(self._palette)
        color_rgb = self._palette[color_idx]

        polygon = det.get("polygon", [])
        if not polygon and "mask" in det:
            from core.interactive_segmentation import mask_to_polygon
            polygon = mask_to_polygon(det["mask"], tolerance=1.5)

        track = CoralTrack(
            track_id=track_id,
            bbox=det.get("bbox", [0, 0, 10, 10]),
            polygon=polygon,
            area_px=det.get("area_px", 0),
            area_pct=det.get("area_pct", 0.0),
            predicted_iou=det.get("predicted_iou", 0.85),
            centroid=det.get("centroid", (0, 0)),
            color_rgb=color_rgb,
            frame_idx=frame_idx,
            taxonomy=det.get("taxonomy"),
            bleaching=det.get("bleaching"),
            mask=det.get("segmentation", det.get("mask")),
        )
        self._tracks[track_id] = track
        return track
