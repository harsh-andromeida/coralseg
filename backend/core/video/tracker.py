"""
Benthic Coral Colony Tracker with Persistent Global IDs and IoU Association.
Maintains persistent track IDs across underwater video frames.
Eliminates dense mask duplication in memory, enforces canonical detection geometry,
bounds history to active window, and handles temporal retirement with a 2-second grace period.
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
    """
    Represents a single tracked coral colony with persistent ID, canonical geometry,
    and compact metadata.
    """

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
        compressed_mask: Optional[bytes] = None,
        max_history_len: int = 30,
    ):
        self.track_id = int(track_id)
        self.bbox = [int(v) for v in bbox]  # [x, y, w, h] in original-space pixels
        self.polygon = [[float(p[0]), float(p[1])] for p in polygon]
        self.area_px = int(area_px)
        self.area_pct = float(area_pct)
        self.predicted_iou = float(predicted_iou)
        self.centroid = (int(centroid[0]), int(centroid[1]))
        self.color_rgb = tuple(int(c) for c in color_rgb)
        self.color_hex = f"#{self.color_rgb[0]:02x}{self.color_rgb[1]:02x}{self.color_rgb[2]:02x}"
        self.first_seen_frame = int(frame_idx)
        self.last_seen_frame = int(frame_idx)
        self.compressed_mask = compressed_mask
        self.max_history_len = max_history_len

        # Default placeholder taxonomy and bleaching until evaluated
        self.taxonomy = dict(taxonomy) if taxonomy else {
            "genus": "Coral",
            "primary_label": f"Coral Colony #{track_id}",
            "growth_form": "Encrusting / Massive",
            "confidence": 0.85,
            "confidence_pct": 85.0,
            "color": self.color_rgb,
        }
        self.bleaching = dict(bleaching) if bleaching else {
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
        # Bounded history window to avoid unbounded RAM accumulation
        self.history: List[Tuple[List[int], List[List[float]]]] = [(list(self.bbox), list(self.polygon))]

    def update_geometry(
        self,
        bbox: List[int],
        polygon: List[List[float]],
        centroid: Tuple[int, int],
        area_px: int,
        area_pct: float,
        frame_idx: int,
        predicted_iou: Optional[float] = None,
        compressed_mask: Optional[bytes] = None,
    ):
        """Updates track geometry and appends to bounded history."""
        self.bbox = [int(v) for v in bbox]
        self.polygon = [[float(p[0]), float(p[1])] for p in polygon]
        self.centroid = (int(centroid[0]), int(centroid[1]))
        self.area_px = int(area_px)
        self.area_pct = float(area_pct)
        self.last_seen_frame = int(frame_idx)
        self.misses = 0
        self.hits += 1
        self.active = True
        if predicted_iou is not None:
            self.predicted_iou = float(predicted_iou)
        if compressed_mask is not None:
            self.compressed_mask = compressed_mask

        self.history.append((list(self.bbox), list(self.polygon)))
        if len(self.history) > self.max_history_len:
            self.history.pop(0)

    def snapshot(self) -> "CoralTrack":
        """
        Returns a lightweight snapshot of this track at the current instant.
        Stores compact geometry and metadata without dense mask copying.
        """
        snap = CoralTrack(
            track_id=self.track_id,
            bbox=list(self.bbox),
            polygon=[list(pt) for pt in self.polygon],
            area_px=self.area_px,
            area_pct=self.area_pct,
            predicted_iou=self.predicted_iou,
            centroid=self.centroid,
            color_rgb=self.color_rgb,
            frame_idx=self.last_seen_frame,
            taxonomy=dict(self.taxonomy) if self.taxonomy else None,
            bleaching=dict(self.bleaching) if self.bleaching else None,
            compressed_mask=self.compressed_mask,
            max_history_len=self.max_history_len,
        )
        snap.first_seen_frame = self.first_seen_frame
        snap.last_seen_frame = self.last_seen_frame
        snap.hits = self.hits
        snap.misses = self.misses
        snap.active = self.active
        return snap

    def to_dict(self) -> Dict[str, Any]:
        """Serializes track state for API responses and video overlay rendering."""
        return {
            "track_id": self.track_id,
            "id": self.track_id,
            "id_str": f"#{self.track_id}",
            "bbox": self.bbox,
            "polygon": self.polygon,
            "centroid": list(self.centroid),
            "area_px": self.area_px,
            "area": self.area_px,
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
            "color_rgb": list(self.color_rgb),
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
    Tracks coral colonies across video frames using IoU spatial association on keyframes.
    Enforces canonical detection geometry, 2-second temporal grace period for absent colonies,
    and supports deterministic global ID reconciliation.
    """

    def __init__(
        self,
        iou_match_threshold: float = 0.30,
        grace_period_sec: float = 2.0,
        fps: float = 25.0,
        min_hits_to_confirm: int = 1,
    ):
        self.iou_match_threshold = float(iou_match_threshold)
        self.grace_period_sec = float(grace_period_sec)
        self.fps = float(fps)
        self.grace_frames = max(10, int(round(self.grace_period_sec * self.fps)))
        self.min_hits_to_confirm = int(min_hits_to_confirm)

        self._next_track_id = 1
        self._tracks: Dict[int, CoralTrack] = {}
        self._palette = generate_distinct_colors(200)

    def reset(self):
        """Resets all active and historical tracks."""
        self._next_track_id = 1
        self._tracks.clear()

    def get_active_tracks(self) -> List[CoralTrack]:
        """Returns all currently active tracks within viewport."""
        return [t for t in self._tracks.values() if t.active]

    def get_all_tracks(self) -> List[CoralTrack]:
        """Returns all known tracks including retired ones."""
        return list(self._tracks.values())

    def get_tracks_needing_enrichment(self) -> List[CoralTrack]:
        """Returns tracks that have not yet undergone BioCLIP / YOLO Bleaching."""
        return [t for t in self._tracks.values() if t.active and t.needs_enrichment]

    def retire_absent_tracks(self, current_frame_idx: int) -> List[int]:
        """
        Retires tracks that have been absent for longer than the grace period.
        Returns list of newly retired track IDs so their inference workload can be purged.
        """
        retired_ids = []
        for t in self._tracks.values():
            if t.active and (current_frame_idx - t.last_seen_frame > self.grace_frames):
                t.active = False
                retired_ids.append(t.track_id)
        return retired_ids

    def match_keyframe_detections(
        self,
        new_detections: List[Dict[str, Any]],
        frame_idx: int,
        frame_w: int,
        frame_h: int,
    ) -> List[CoralTrack]:
        """
        Associates freshly segmented SAM masks on a Keyframe with active tracks.
        Updates matched tracks with refined boundaries; spawns new tracks for newly uncovered corals.
        Normalizes area vs area_px and recomputes percentages after scaling.
        """
        active_tracks = [t for t in self._tracks.values() if t.active]
        total_frame_px = max(1, frame_w * frame_h)

        # Normalize detections to canonical representation at the producer boundary
        normalized_dets = []
        for det in new_detections:
            raw_area = det.get("area_px", det.get("area", 0))
            area_px = int(round(raw_area))
            area_pct = round((area_px / total_frame_px) * 100.0, 4)

            bbox = det.get("bbox", [0, 0, 1, 1])
            polygon = det.get("polygon", [])
            if not polygon and "mask" in det:
                from core.interactive_segmentation import mask_to_polygon
                polygon = mask_to_polygon(det["mask"], tolerance=1.5)

            centroid = det.get("centroid")
            if not centroid:
                centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            d = dict(det)
            d["area_px"] = area_px
            d["area"] = area_px
            d["area_pct"] = area_pct
            d["bbox"] = bbox
            d["polygon"] = polygon
            d["centroid"] = centroid
            normalized_dets.append(d)

        n_tracks = len(active_tracks)
        n_dets = len(normalized_dets)

        if n_tracks == 0:
            for det in normalized_dets:
                self._spawn_track(det, frame_idx, frame_w, frame_h)
            return self.get_active_tracks()

        if n_dets == 0:
            self.retire_absent_tracks(frame_idx)
            return self.get_active_tracks()

        # Compute IoU + spatial proximity score matrix
        score_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, track in enumerate(active_tracks):
            tcx, tcy = track.centroid
            t_diag = float(np.sqrt(max(track.bbox[2]**2 + track.bbox[3]**2, 100.0)))
            for j, det in enumerate(normalized_dets):
                det_bbox = det["bbox"]
                iou = compute_box_iou(track.bbox, det_bbox)

                dcx, dcy = det["centroid"]
                dist = float(np.sqrt((tcx - dcx)**2 + (tcy - dcy)**2))
                norm_dist = dist / max(t_diag, 10.0)
                center_sim = max(0.0, 1.0 - min(norm_dist, 1.0))

                score = 0.65 * iou + 0.35 * center_sim
                if track.taxonomy.get("genus") == det.get("taxonomy", {}).get("genus"):
                    score += 0.05
                score_matrix[i, j] = score

        # Greedy bipartite matching
        matched_tracks = set()
        matched_dets = set()

        pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                det_bbox = normalized_dets[j]["bbox"]
                iou = compute_box_iou(active_tracks[i].bbox, det_bbox)
                if score_matrix[i, j] >= self.iou_match_threshold or iou >= self.iou_match_threshold:
                    pairs.append((score_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, trk_idx, det_idx in pairs:
            if trk_idx in matched_tracks or det_idx in matched_dets:
                continue

            matched_tracks.add(trk_idx)
            matched_dets.add(det_idx)

            track = active_tracks[trk_idx]
            det = normalized_dets[det_idx]

            track.update_geometry(
                bbox=det["bbox"],
                polygon=det["polygon"],
                centroid=det["centroid"],
                area_px=det["area_px"],
                area_pct=det["area_pct"],
                frame_idx=frame_idx,
                predicted_iou=det.get("predicted_iou"),
            )

            if "taxonomy" in det and det["taxonomy"]:
                track.taxonomy = det["taxonomy"]
            if "bleaching" in det and det["bleaching"]:
                track.bleaching = det["bleaching"]

        # Check absent tracks for retirement
        self.retire_absent_tracks(frame_idx)

        # Spawn new tracks for unmatched detections
        for j, det in enumerate(normalized_dets):
            if j not in matched_dets:
                self._spawn_track(det, frame_idx, frame_w, frame_h)

        return self.get_active_tracks()

    def _spawn_track(
        self,
        det: Dict[str, Any],
        frame_idx: int,
        frame_w: int,
        frame_h: int,
        explicit_id: Optional[int] = None,
    ) -> CoralTrack:
        """
        Spawns a new persistent CoralTrack. If explicit_id is provided,
        does not increment unique colony counts past max(existing, explicit_id + 1).
        """
        if explicit_id is not None:
            track_id = explicit_id
            if track_id >= self._next_track_id:
                self._next_track_id = track_id + 1
        else:
            track_id = self._next_track_id
            self._next_track_id += 1

        color_idx = (track_id - 1) % len(self._palette)
        color_rgb = self._palette[color_idx]

        total_frame_px = max(1, frame_w * frame_h)
        raw_area = det.get("area_px", det.get("area", 0))
        area_px = int(round(raw_area))
        area_pct = round((area_px / total_frame_px) * 100.0, 4)

        bbox = det.get("bbox", [0, 0, 10, 10])
        polygon = det.get("polygon", [])
        centroid = det.get("centroid", (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))

        track = CoralTrack(
            track_id=track_id,
            bbox=bbox,
            polygon=polygon,
            area_px=area_px,
            area_pct=area_pct,
            predicted_iou=det.get("predicted_iou", 0.85),
            centroid=centroid,
            color_rgb=color_rgb,
            frame_idx=frame_idx,
            taxonomy=det.get("taxonomy"),
            bleaching=det.get("bleaching"),
        )
        self._tracks[track_id] = track
        return track

    def map_local_id_to_global(self, local_id: int, global_id: int):
        """
        Reconciles a chunk-local ID to an existing global ID.
        Ensures global ID counts are not incremented spuriously.
        """
        if local_id in self._tracks and local_id != global_id:
            track = self._tracks.pop(local_id)
            track.track_id = global_id
            self._tracks[global_id] = track
            if global_id >= self._next_track_id:
                self._next_track_id = global_id + 1

    @property
    def total_unique_colonies(self) -> int:
        """Returns count of all unique colonies cataloged so far."""
        return len(self._tracks)
