from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class DevicePreferenceRequest(BaseModel):
    preference: str = Field(default="auto", description="Device preference: auto, cuda, cpu")
    concurrency: Optional[int] = Field(default=1, ge=1, le=4, description="Parallel batch concurrency workers (1-4)")


class DeviceInfoResponse(BaseModel):
    device: str
    device_type: str
    cuda_available: bool
    dml_available: bool
    gpu_name: str
    active_provider: str


class ModelSpec(BaseModel):
    filename: str
    name: str
    task: str
    size: str
    downloaded: bool
    cached: bool
    approx_size: int
    actual_size: int
    path: Optional[str] = None


class ModelStatusResponse(BaseModel):
    all_downloaded: bool
    models: List[ModelSpec]
    active_device: str
    device_info: Dict[str, Any]


class ModelDownloadRequest(BaseModel):
    force: bool = Field(default=False, description="Force re-download from Hugging Face even if cached")
    filename: Optional[str] = Field(default=None, description="Specific model filename or group key")
    filenames: Optional[List[str]] = Field(default=None, description="List of model filenames or group keys")


class ModelDeleteRequest(BaseModel):
    filename: Optional[str] = Field(default=None, description="Specific model filename or group key to delete")
    filenames: Optional[List[str]] = Field(default=None, description="List of filenames or group keys to delete")
    all: bool = Field(default=False, description="Whether to delete all models")


class ModelDeleteResponse(BaseModel):
    status: str
    message: str
    deleted_files: List[str]
    freed_bytes: int
    all_downloaded: bool
    models: List[ModelSpec]


class SegmentRequest(BaseModel):
    image_name: str
    image_base64: Optional[str] = None
    points_per_side: int = Field(default=16, ge=8, le=36)
    iou_thresh: float = Field(default=0.50, ge=0.20, le=0.98)
    stability_thresh: float = Field(default=0.50, ge=0.20, le=0.99)


class EnrichRequest(BaseModel):
    image_name: str
    min_area_px: int = Field(default=100, ge=10, le=50000)


class OverlayRequest(BaseModel):
    image_name: str
    min_area_px: int = Field(default=100, ge=10, le=50000)
    alpha: float = Field(default=0.45, ge=0.10, le=0.90)
    draw_contours: bool = True
    draw_labels: bool = True
    draw_boxes: bool = False
    selected_mask_id: Optional[int] = None
    color_mode: str = Field(default="instance", description="instance | bleaching | taxonomy | taxonomy_condition")
    layout_mode: str = Field(default="Side-by-Side", description="Side-by-Side | Overlay Only | Original Only | Masks on Black")


class ExportCocoRequest(BaseModel):
    image_name: str
    min_area_px: int = Field(default=100, ge=10, le=50000)


class SampleItem(BaseModel):
    filename: str
    path: str
    size_bytes: int


class BatchStartRequest(BaseModel):
    images: List[str] = Field(description="List of registered image names to process in bulk")
    concurrency: int = Field(default=1, ge=1, le=4, description="Parallel worker count (1: sequential, 2-4: parallel)")
    points_per_side: int = Field(default=16, ge=8, le=36)
    iou_thresh: float = Field(default=0.50, ge=0.20, le=0.98)
    stability_thresh: float = Field(default=0.50, ge=0.20, le=0.99)
    min_area_px: int = Field(default=100, ge=10, le=50000)


class BatchPrioritizeRequest(BaseModel):
    image_name: str = Field(description="Name of image to move to front of batch queue")


class BatchStatusResponse(BaseModel):
    batch_id: str
    is_running: bool
    is_paused: bool
    concurrency: int
    total: int
    completed: int
    failed: int
    percent: int
    current_image: Optional[str]
    current_stage: str
    images_order: List[str]
    items: Dict[str, Any]


class ExportFolderRequest(BaseModel):
    folder_path: str = Field(description="Target directory path on local filesystem")
    include_zip: bool = Field(default=False, description="Whether to also generate .zip archive inside the folder")
    open_folder: bool = Field(default=True, description="Whether to open the target folder in OS file manager")
    image_names: Optional[List[str]] = Field(default=None, description="Optional subset of image names to export")


class ExportFolderResponse(BaseModel):
    success: bool
    folder_path: str
    total_images: int
    images_saved: int
    total_corals: int
    written_files_count: int
    zip_file: Optional[str] = None
    error: Optional[str] = None


class PromptPoint(BaseModel):
    x: float
    y: float
    label: int = Field(default=1, description="1: positive / foreground, 0: negative / background")


class SegmentPromptRequest(BaseModel):
    image_name: str
    points: List[PromptPoint]
    image_data: Optional[str] = Field(default=None, description="Optional base64 dataUrl of image")


class SegmentPromptWarmupRequest(BaseModel):
    image_name: str
    image_data: Optional[str] = Field(default=None, description="Optional base64 dataUrl of image")



class EditedSegmentItem(BaseModel):
    id: int
    genus: str = "Coral"
    growth_form: str = "-"
    condition: str = "Healthy"
    color_hex: Optional[str] = None
    polygon: List[List[float]] = Field(description="List of [x, y] polygon coordinates")


class SaveAnnotationsRequest(BaseModel):
    image_name: str
    segments: List[EditedSegmentItem]
    image_data: Optional[str] = Field(default=None, description="Optional base64 dataUrl of image")


# --- Video Processing Schemas ---

class VideoProcessRequest(BaseModel):
    video_name: str = Field(description="Filename of registered video in uploads")
    overlap_threshold: float = Field(default=0.75, ge=0.40, le=0.95, description="Scene overlap threshold triggering a keyframe")
    min_stride: int = Field(default=15, ge=1, le=60, description="Minimum frame spacing between keyframes")
    max_stride: int = Field(default=45, ge=5, le=120, description="Maximum frame spacing between keyframes")
    points_per_side: int = Field(default=16, ge=8, le=32, description="SAM prompt grid density for keyframes")
    iou_match_threshold: float = Field(default=0.30, ge=0.10, le=0.80, description="IoU threshold for tracking association")
    pred_iou_thresh: float = Field(default=0.50, ge=0.20, le=0.90, description="SAM predicted IoU threshold")
    stability_thresh: float = Field(default=0.50, ge=0.20, le=0.90, description="SAM stability score threshold")
    tracker_mode: Optional[str] = Field(default="sam2", description="Compatibility tracker mode ('sam2')")
    model_profile: str = Field(default="sam2", description="Video tracker profile: 'sam2' (default reference) or 'efficienttam' (fast profile)")
    model_size: int = Field(default=1024, description="Model input resolution: 1024 (reference) or 512 (fast)")
    precision: Optional[str] = Field(default=None, description="Precision: 'float16', 'bfloat16', 'float32', or auto (None)")
    worker_count: Union[int, str] = Field(default="auto", description="Inference worker instances: 'auto' or 1-8")
    export_mp4: bool = Field(default=True, description="Whether to render annotated MP4 video deliverable")

    @field_validator("tracker_mode")
    @classmethod
    def validate_tracker_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_clean = v.strip().lower()
            if v_clean in ("optical_flow", "flow", "lucas_kanade"):
                raise ValueError(
                    "Optical flow inference engine has been removed. "
                    "Use 'sam2' (default reference) or 'efficienttam' (experimental fast profile)."
                )
            if v_clean not in ("sam2", "efficienttam"):
                raise ValueError(f"Unsupported tracker mode '{v}'. Allowed: 'sam2', 'efficienttam'")
            return v_clean
        return "sam2"

    @field_validator("model_profile")
    @classmethod
    def validate_model_profile(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean not in ("sam2", "efficienttam"):
            raise ValueError(f"Unsupported model profile '{v}'. Allowed: 'sam2', 'efficienttam'")
        return v_clean

    @field_validator("model_size")
    @classmethod
    def validate_model_size(cls, v: int) -> int:
        if v not in (512, 1024):
            raise ValueError(f"Unsupported model_size {v}. Allowed: 512, 1024")
        return v

    @model_validator(mode="after")
    def validate_strides(self) -> "VideoProcessRequest":
        if self.min_stride > self.max_stride:
            raise ValueError(f"min_stride ({self.min_stride}) cannot exceed max_stride ({self.max_stride})")
        return self


class VideoTrackItem(BaseModel):
    track_id: int
    bbox: List[int]
    polygon: List[List[float]]
    genus: str = "Coral"
    primary_label: str = "Coral"
    growth_form: str = "-"
    taxon_conf: float = 85.0
    condition: str = "Healthy"
    condition_conf: float = 90.0
    is_bleached: bool = False
    color_hex: str = "#00ffff"


class VideoStatusResponse(BaseModel):
    job_id: str
    filename: str
    state: str
    current_frame: int
    total_frames: int
    progress_pct: float
    fps_realtime: float
    tracker_fps: float = 0.0
    analysis_fps: float = 0.0
    elapsed_sec: float = 0.0
    active_corals_count: int
    total_unique_corals: int
    bleaching_prevalence_pct: float
    keyframes_count: int
    metadata: Dict[str, Any]
    error: Optional[str] = None
    time_series: Optional[List[Dict[str, Any]]] = None
    export_paths: Optional[Dict[str, Optional[str]]] = None
    export_status: Optional[str] = "pending"
    current_stage: Optional[str] = "Processing"
    current_model: Optional[str] = None
    current_action: Optional[str] = None
    stage_detail: Optional[str] = None
    logs: Optional[List[str]] = None
    tracker_mode: Optional[str] = "sam2"



