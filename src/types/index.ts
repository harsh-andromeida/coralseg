export interface DeviceInfo {
  device: string;
  device_type: "gpu" | "cpu";
  device_name?: string;
  native_precision?: string;
  compute_capability?: string | null;
  mps_available?: boolean;
  coreml_available?: boolean;
  mode?: "gpu" | "multithread_cpu" | "cpu";
  mode_label?: string;
  cuda_available: boolean;
  torch_cuda_available?: boolean;
  ort_cuda_available?: boolean;
  cuda_status?: "active" | "onnxruntime_gpu_missing" | "unavailable" | string;
  cuda_diagnostic?: string | null;
  cuda_fix_command?: string | null;
  dml_available: boolean;
  gpu_available?: boolean;
  gpu_name: string;
  active_provider: string;
  threads?: number;
  cpu_count?: number;
  preference?: string;
  priority?: string[];
  concurrency?: number;
}

export interface ModelSpec {
  filename: string;
  name: string;
  task: string;
  size: string;
  downloaded: boolean;
  cached: boolean;
  approx_size: number;
  actual_size: number;
  path?: string;
}

export interface ModelStatusResponse {
  all_downloaded: boolean;
  models: ModelSpec[];
  active_device: string;
  device_info: DeviceInfo;
}

export interface ModelDeleteResponse {
  status: string;
  message: string;
  deleted_files: string[];
  freed_bytes: number;
  all_downloaded: boolean;
  models: ModelSpec[];
}

export interface DownloadProgress {
  is_downloading: boolean;
  overall_pct: number;
  current_file: string;
  detail: string;
  error?: string;
}

export interface SampleItem {
  filename: string;
  path: string;
  size_bytes: number;
}

export interface CoralSegment {
  id: number;
  id_str: string;
  genus: string;
  growth_form: string;
  taxon_conf: number;
  condition: "Healthy" | "Bleached";
  condition_conf: number;
  area_pct: number;
  area_px: number;
  predicted_iou: number;
  color_hex: string;
  centroid?: [number, number];
  bbox?: [number, number, number, number];
  polygon?: [number, number][];
  is_human_annotated?: boolean;
}

export interface PromptPoint {
  x: number;
  y: number;
  label: number; // 1: foreground, 0: background
}

export interface PromptSegmentResponse {
  status: string;
  image_name: string;
  polygon: [number, number][];
  area_px: number;
  area_pct: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  predicted_iou: number;
}

export interface SaveAnnotationsResponse {
  status: string;
  image_name: string;
  summary: SummaryStats;
  segments: CoralSegment[];
  health_summary: HealthSummary;
  overlay_base64: string;
}


export interface SummaryStats {
  total_corals_detected: number;
  coral_coverage_pct: number;
  coral_covered_pixels: number;
  total_image_pixels: number;
  image_resolution: string;
  mean_iou_confidence: number;
  mean_stability_score: number;
}

export interface HealthSummary {
  healthy_count: number;
  bleached_count: number;
  bleaching_prevalence_pct: number;
}

export interface AnalysisData {
  summary: SummaryStats;
  segments: CoralSegment[];
  health_summary: HealthSummary;
  scene_eval: Record<string, any>;
}

export interface LoadedImage {
  name: string;
  dataUrl: string;
  width?: number;
  height?: number;
}

export type AnalysisStage = "segmenting" | "classifying" | "rendering" | "idle" | "pending";

export interface BatchItemStatus {
  image_name: string;
  status: "pending" | "processing" | "completed" | "error" | "cancelled";
  stage: string;
  error?: string | null;
  corals_count?: number;
  coverage_pct?: number;
  bleaching_prevalence_pct?: number;
  summary?: SummaryStats | null;
  segments?: CoralSegment[] | null;
  health_summary?: HealthSummary | null;
  scene_eval?: Record<string, any> | null;
  overlay_base64?: string | null;
}

export interface BatchStatusResponse {
  batch_id: string;
  is_running: boolean;
  is_paused: boolean;
  concurrency: number;
  total: number;
  completed: number;
  failed: number;
  percent: number;
  current_image: string | null;
  current_stage: string;
  images_order: string[];
  items: Record<string, BatchItemStatus>;
}

export interface DefaultExportPathResponse {
  default_folder: string;
  parent_dir: string;
  suggested_name: string;
}

export interface ExportToFolderRequest {
  folder_path: string;
  include_zip?: boolean;
  open_folder?: boolean;
  image_names?: string[];
}

export interface ExportToFolderResponse {
  success: boolean;
  folder_path: string;
  total_images: number;
  images_saved: number;
  total_corals: number;
  written_files_count: number;
  zip_file?: string | null;
  error?: string | null;
}

export type ExportFolderResponse = ExportToFolderResponse;

// --- Video Processing & Benthic Tracking Interfaces ---

export interface VideoMetadata {
  width: number;
  height: number;
  fps: number;
  total_frames: number;
  duration_sec: number;
  filename: string;
  file_size: number;
}

export interface VideoTrackItem {
  track_id: number;
  id: number;
  id_str: string;
  bbox: [number, number, number, number];
  polygon: [number, number][];
  centroid?: [number, number];
  area_px: number;
  area_pct: number;
  predicted_iou: number;
  genus: string;
  primary_label: string;
  growth_form: string;
  taxon_conf: number;
  condition: "Healthy" | "Bleached";
  condition_conf: number;
  is_bleached: boolean;
  bleached_prob?: number;
  color_rgb?: [number, number, number];
  color_hex: string;
  first_seen_frame: number;
  last_seen_frame: number;
  hits?: number;
  active?: boolean;
}

export interface TransectMetricPoint {
  frame_index: number;
  timestamp_sec: number;
  active_corals_count: number;
  cumulative_unique_corals: number;
  bleached_corals_count: number;
  healthy_corals_count: number;
  bleaching_prevalence_pct: number;
  dominant_genus: string;
  is_keyframe: boolean;
  camera_motion_dx: number;
  camera_motion_dy: number;
}

export interface VideoJobStatus {
  job_id: string;
  filename: string;
  state: "queued" | "preparing" | "processing" | "finalizing" | "completed" | "cancelling" | "cancelled" | "error" | "paused";
  current_frame: number;
  total_frames: number;
  progress_pct: number;
  fps_realtime: number;
  tracker_fps?: number;
  analysis_fps?: number;
  elapsed_sec?: number;
  active_corals_count: number;
  total_unique_corals: number;
  bleaching_prevalence_pct: number;
  keyframes_count: number;
  metadata: VideoMetadata;
  error?: string | null;
  time_series?: TransectMetricPoint[];
  export_paths?: {
    annotated_mp4?: string | null;
    transect_csv?: string | null;
    coco_json?: string | null;
  };
  export_status?: "pending" | "ready" | "failed";
  current_stage?: string;
  current_model?: string;
  current_action?: string;
  stage_detail?: string;
  stage_progress_pct?: number;
  logs?: string[];
  tracker_mode?: string;
  model_profile?: "sam2" | "efficienttam";
  model_size?: 512 | 1024;
  worker_count?: number;
}

export interface VideoProcessParams {
  video_name: string;
  overlap_threshold?: number;
  min_stride?: number;
  max_stride?: number;
  points_per_side?: number;
  iou_match_threshold?: number;
  pred_iou_thresh?: number;
  stability_thresh?: number;
  model_profile?: "sam2" | "efficienttam";
  model_size?: 512 | 1024;
  worker_count?: number;
  precision?: string | null;
  export_mp4?: boolean;
}

export interface UploadedVideoItem {
  file: File;
  video_name: string;
  video_path?: string;
  metadata: VideoMetadata;
  previewUrl: string;
  job_id?: string;
}



