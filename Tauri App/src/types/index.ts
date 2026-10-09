export interface DeviceInfo {
  device: string;
  device_type: "gpu" | "cpu";
  mode?: "gpu" | "multithread_cpu" | "cpu";
  mode_label?: string;
  cuda_available: boolean;
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


