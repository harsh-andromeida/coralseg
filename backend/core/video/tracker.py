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
        global_colony_id: Optional[int] = None,
    ):
        self.track_id = int(track_id)
        self.global_colony_id = int(global_colony_id) if global_colony_id is not None else int(track_id)
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
            global_colony_id=self.global_colony_id,
        )
        snap.first_seen_frame = self.first_seen_frame
        snap.last_seen_frame = self.last_seen_frame
        snap.hits = self.hits
        snap.misses = self.misses
        snap.active = self.active
        return snap

    def to_dict(self) -> Dict[str, Any]:
        """Serializes track state for API responses and video overlay rendering."""
        gid = self.global_colony_id
        return {
            "track_id": gid,
            "id": gid,
            "id_str": f"#{gid}",
            "local_track_id": self.track_id,
            "global_colony_id": gid,
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
