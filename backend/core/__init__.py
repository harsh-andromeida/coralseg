"""
Core engine package for CoralSCOP dense instance segmentation, BioCLIP taxonomy, and NOAA bleaching analysis.
"""

from core.device import (
    get_optimal_device,
    get_device_info,
    get_device_mode,
    get_session_options,
    resolve_hardware_tier,
)
from core.models import (
    HF_REPO_ID,
    HF_SAM_DECODER_FILE,
    HF_SAM_ENCODER_FILE,
    HF_BIOCLIP_FILE,
    HF_BLEACHING_FILE,
    HF_TEXT_EMBEDS_FILE,
    EXPECTED_MODELS,
    get_cached_model_path,
    find_or_download_onnx_model,
    check_models_download_status,
    download_all_models,
    download_coralscop_checkpoint,
    load_coralscop_model,
    load_bioclip_model,
    load_bleaching_model,
)
from core.segmentation import (
    run_segmentation,
    calculate_stability_score,
    nms_boxes,
    mask_nms,
)
from core.taxonomy import (
    GENUS_COLORS,
    get_genus_color,
    CORAL_TAXONOMY_CANDIDATES,
    classify_coral_taxonomy,
    detect_coral_bleaching,
    enrich_masks_with_taxonomy_and_bleaching,
)
from core.visualization import (
    generate_distinct_colors,
    get_overlay_font,
    get_mask_overlay_color,
    get_mask_overlay_label,
    create_segmentation_overlay,
)
from core.export import build_coco_json

__all__ = [
    "get_optimal_device",
    "get_device_info",
    "get_device_mode",
    "get_session_options",
    "resolve_hardware_tier",
    "HF_REPO_ID",
    "HF_SAM_DECODER_FILE",
    "HF_SAM_ENCODER_FILE",
    "HF_BIOCLIP_FILE",
    "HF_BLEACHING_FILE",
    "HF_TEXT_EMBEDS_FILE",
    "EXPECTED_MODELS",
    "get_cached_model_path",
    "find_or_download_onnx_model",
    "check_models_download_status",
    "download_all_models",
    "download_coralscop_checkpoint",
    "load_coralscop_model",
    "load_bioclip_model",
    "load_bleaching_model",
    "run_segmentation",
    "calculate_stability_score",
    "nms_boxes",
    "mask_nms",
    "GENUS_COLORS",
    "get_genus_color",
    "CORAL_TAXONOMY_CANDIDATES",
    "classify_coral_taxonomy",
    "detect_coral_bleaching",
    "enrich_masks_with_taxonomy_and_bleaching",
    "generate_distinct_colors",
    "get_overlay_font",
    "get_mask_overlay_color",
    "get_mask_overlay_label",
    "create_segmentation_overlay",
    "build_coco_json",
]
