"""
Canonical shapes of the plain dicts that flow through the image pipeline.

These are TypedDicts: zero runtime cost, no behaviour change. They exist so a reader
(human or agent) can see every key once instead of reverse-engineering producers.

Lifecycle of a segment dict:
  1. `core.segmentation.run_segmentation`           -> `Segment` (base keys)
  2. `core.taxonomy.enrich_masks_with_taxonomy_...` -> `EnrichedSegment` (adds taxonomy/bleaching/colours)
  3. `app.routers.images.save_manual_annotations`   -> `EnrichedSegment` with `is_human_annotated=True`
  4. Routers/batch convert to `SegmentRow` (JSON-safe table record, no masks) for the UI.

Duplicate keys exist for historical reasons and are all populated; prefer the first of each pair:
  id / segment_id, mask / segmentation, area_px / area, color_rgb / instance_color_rgb (after enrichment).
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np


class Segment(TypedDict):
    """Raw SAM instance as returned by `run_segmentation`."""
    id: int                          # 1-based rank
    segment_id: int                  # duplicate of id
    mask: np.ndarray                 # (H, W) bool, full image resolution
    segmentation: np.ndarray         # duplicate of mask
    area_px: int
    area: int                        # duplicate of area_px
    area_pct: float                  # percent of image pixels
    bbox: List[int]                  # [x, y, w, h]
    predicted_iou: float
    stability_score: float
    centroid: Tuple[int, int]        # (cx, cy)
    color_rgb: Tuple[int, int, int]
    color_hex: str


class Taxonomy(TypedDict, total=False):
    genus: str
    growth_form: str
    primary_label: str
    confidence_pct: float
    color: Tuple[int, int, int]


class Bleaching(TypedDict, total=False):
    is_bleached: bool
    status: str                      # "Healthy" | "Bleached"
    confidence_pct: float
    bleached_prob: float


class EnrichedSegment(Segment, total=False):
    """Segment after BioCLIP/bleaching enrichment or manual annotation."""
    taxonomy: Taxonomy
    bleaching: Bleaching
    instance_color_rgb: Tuple[int, int, int]
    instance_color_hex: str
    taxonomy_color_rgb: Tuple[int, int, int]
    taxonomy_color_hex: str
    color: Tuple[int, int, int]
    polygon: List[List[float]]       # only for manual annotations
    is_human_annotated: bool


class SegmentRow(TypedDict, total=False):
    """JSON-safe table record sent to the UI (no mask arrays)."""
    id: int
    id_str: str
    genus: str
    growth_form: str
    taxon_conf: float
    condition: str
    condition_conf: float
    area_pct: float
    area_px: int
    predicted_iou: float
    color_hex: str
    centroid: Optional[List[int]]
    bbox: Optional[List[int]]
    is_human_annotated: bool


class HealthSummary(TypedDict, total=False):
    healthy_count: int
    bleached_count: int
    bleaching_prevalence_pct: float


# Cache value shapes (see app/state.py)
BaseSegCache = Dict[str, List[Segment]]
EnrichCache = Dict[str, Tuple[List[EnrichedSegment], Dict[str, Any], HealthSummary]]
