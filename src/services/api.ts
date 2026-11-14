import {
  DeviceInfo,
  ModelStatusResponse,
  ModelDeleteResponse,
  DownloadProgress,
  SampleItem,
  AnalysisData,
  BatchStatusResponse,
  DefaultExportPathResponse,
  ExportToFolderRequest,
  ExportFolderResponse,
  CoralSegment,
  PromptPoint,
  PromptSegmentResponse,
  SaveAnnotationsResponse,
} from "../types";

export function getApiBase(): string {
  if (typeof window !== "undefined") {
    const custom = localStorage.getItem("REEFVISION_API_URL");
    if (custom && custom.trim()) {
      const clean = custom.trim().replace(/\/+$/, "");
      return clean.endsWith("/api") ? clean : `${clean}/api`;
    }

    // If running inside native packaged Tauri desktop app where origin is tauri:// or tauri.localhost
    const isTauriScheme =
      window.location.protocol === "tauri:" ||
      window.location.hostname === "tauri.localhost" ||
      ("__TAURI_INTERNALS__" in window && window.location.port !== "1425");
    if (isTauriScheme) {
      return "http://127.0.0.1:8000/api";
    }
  }
  return "/api";
}

export function getCustomApiUrl(): string {
  if (typeof window !== "undefined") {
    return localStorage.getItem("REEFVISION_API_URL") || "";
  }
  return "";
}

export function setCustomApiUrl(url: string | null) {
  if (typeof window !== "undefined") {
    if (url && url.trim()) {
      localStorage.setItem("REEFVISION_API_URL", url.trim().replace(/\/+$/, ""));
    } else {
      localStorage.removeItem("REEFVISION_API_URL");
    }
  }
}

const API_BASE = {
  toString: () => getApiBase(),
};

export async function checkHealth(): Promise<{ status: string; device: DeviceInfo }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Backend offline");
  return res.json();
}

export async function getDevice(): Promise<DeviceInfo> {
  const res = await fetch(`${API_BASE}/device/info`);
  if (!res.ok) throw new Error("Failed to fetch device info");
  return res.json();
}

export async function getModelsStatus(): Promise<ModelStatusResponse> {
  const res = await fetch(`${API_BASE}/models/status`);
  if (!res.ok) throw new Error("Failed to fetch models status");
  return res.json();
}

export async function triggerModelDownload(
  force: boolean = false,
  target?: string | string[]
): Promise<void> {
  const body: any = { force };
  if (Array.isArray(target)) {
    body.filenames = target;
  } else if (target) {
    body.filename = target;
  }
  const res = await fetch(`${API_BASE}/models/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("Failed to start model download");
}

export async function deleteModel(
  target?: string | string[]
): Promise<ModelDeleteResponse> {
  let body: any;
  if (!target) {
    body = { all: true };
  } else if (Array.isArray(target)) {
    body = { filenames: target, all: false };
  } else {
    body = { filename: target, all: false };
  }
  const res = await fetch(`${API_BASE}/models/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to delete model");
  }
  return res.json();
}

export async function getDownloadProgress(): Promise<DownloadProgress> {
  const res = await fetch(`${API_BASE}/models/download/progress`);
  if (!res.ok) throw new Error("Failed to get download progress");
  return res.json();
}

export async function setDevicePreference(
  preference: string,
  concurrency?: number
): Promise<DeviceInfo> {
  const res = await fetch(`${API_BASE}/device/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ preference, concurrency }),
  });
  if (!res.ok) throw new Error("Failed to set device preference");
  return res.json();
}

export async function getSamples(): Promise<SampleItem[]> {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error("Failed to fetch sample images");
  return res.json();
}

export function getSampleImageUrl(filename: string): string {
  return `${API_BASE}/samples/image/${encodeURIComponent(filename)}`;
}

export async function registerImage(filename: string, dataUrl: string): Promise<any> {
  const res = await fetch(`${API_BASE}/images/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, data: dataUrl }),
  });
  if (!res.ok) throw new Error("Failed to register image");
  return res.json();
}

export async function loadSampleToStore(filename: string): Promise<any> {
  const res = await fetch(`${API_BASE}/images/load-sample`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  if (!res.ok) throw new Error("Failed to load sample to store");
  return res.json();
}

export async function runSegmentation(
  imageName: string,
  pointsPerSide: number,
  iouThresh: number,
  stabilityThresh: number,
  imageBase64?: string
): Promise<any> {
  const res = await fetch(`${API_BASE}/analysis/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: imageName,
      points_per_side: pointsPerSide,
      iou_thresh: iouThresh,
      stability_thresh: stabilityThresh,
      image_base64: imageBase64,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Segmentation failed");
  }
  return res.json();
}

export async function runEnrichment(
  imageName: string,
  minAreaPx: number
): Promise<AnalysisData> {
  const res = await fetch(`${API_BASE}/analysis/enrich`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: imageName,
      min_area_px: minAreaPx,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Enrichment failed");
  }
  return res.json();
}

export async function renderOverlay(params: {
  imageName: string;
  minAreaPx: number;
  alpha: number;
  drawContours: boolean;
  drawLabels: boolean;
  drawBoxes: boolean;
  selectedMaskId?: number | null;
  colorMode: string;
  layoutMode: string;
}): Promise<string> {
  const res = await fetch(`${API_BASE}/analysis/overlay`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: params.imageName,
      min_area_px: params.minAreaPx,
      alpha: params.alpha,
      draw_contours: params.drawContours,
      draw_labels: params.drawLabels,
      draw_boxes: params.drawBoxes,
      selected_mask_id: params.selectedMaskId,
      color_mode: params.colorMode,
      layout_mode: params.layoutMode,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Overlay render failed");
  }
  const data = await res.json();
  return data.overlay_base64;
}

export async function exportCocoJson(imageName: string, minAreaPx: number): Promise<any> {
  const res = await fetch(`${API_BASE}/analysis/export/coco`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: imageName,
      min_area_px: minAreaPx,
    }),
  });
  if (!res.ok) throw new Error("COCO export failed");
  return res.json();
}

export async function exportCsvData(imageName: string, minAreaPx: number): Promise<string> {
  const res = await fetch(`${API_BASE}/analysis/export/csv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: imageName,
      min_area_px: minAreaPx,
    }),
  });
  if (!res.ok) throw new Error("CSV export failed");
  return res.text();
}

// -------------------------------------------------------------
// Autonomous Background Batch Processing & Dataset API methods
// -------------------------------------------------------------

export async function startBatch(
  images: string[],
  params?: {
    points_per_side?: number;
    iou_thresh?: number;
    stability_thresh?: number;
    min_area_px?: number;
  },
  concurrency: number = 1
): Promise<BatchStatusResponse> {
  const res = await fetch(`${API_BASE}/batch/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      images,
      concurrency,
      points_per_side: params?.points_per_side ?? 16,
      iou_thresh: params?.iou_thresh ?? 0.50,
      stability_thresh: params?.stability_thresh ?? 0.50,
      min_area_px: params?.min_area_px ?? 100,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start batch processing");
  }
  return res.json();
}

export async function getBatchStatus(): Promise<BatchStatusResponse> {
  const res = await fetch(`${API_BASE}/batch/status`);
  if (!res.ok) throw new Error("Failed to get batch status");
  return res.json();
}

export async function pauseBatch(): Promise<BatchStatusResponse> {
  const res = await fetch(`${API_BASE}/batch/pause`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to pause batch");
  return res.json();
}

export async function resumeBatch(): Promise<BatchStatusResponse> {
  const res = await fetch(`${API_BASE}/batch/resume`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to resume batch");
  return res.json();
}

export async function cancelBatch(): Promise<BatchStatusResponse> {
  const res = await fetch(`${API_BASE}/batch/cancel`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to cancel batch");
  return res.json();
}

export async function unloadModelsFromRam(): Promise<{ unloaded: boolean; message: string }> {
  const res = await fetch(`${API_BASE}/models/unload`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to unload models from RAM");
  return res.json();
}

export async function prioritizeBatchImage(imageName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/batch/prioritize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_name: imageName }),
  });
  if (!res.ok) throw new Error("Failed to prioritize image");
  return res.json();
}

export async function getPrecomputedResult(imageName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/analysis/result/${encodeURIComponent(imageName)}`);
  if (!res.ok) throw new Error("Failed to fetch image analysis result");
  return res.json();
}

export async function getDefaultExportPath(): Promise<DefaultExportPathResponse> {
  const res = await fetch(`${API_BASE}/batch/export/default-path`);
  if (!res.ok) throw new Error("Failed to get default export path");
  return res.json();
}

export async function exportDatasetToFolder(
  params: ExportToFolderRequest
): Promise<ExportFolderResponse> {
  const res = await fetch(`${API_BASE}/batch/export/to-folder`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to export dataset to folder");
  }
  return res.json();
}

export async function pickFolderNative(defaultPath?: string): Promise<string | null> {
  if (typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window)) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return await invoke<string | null>("pick_folder_dialog", {
        defaultPath: defaultPath || null,
      });
    } catch (e) {
      console.warn("Failed to open Tauri native folder dialog:", e);
    }
  }
  return null;
}

export async function openFolderNative(path: string): Promise<void> {
  if (typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window)) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("open_folder", { path });
    } catch (e) {
      console.warn("Failed to open folder natively:", e);
    }
  }
}

export async function downloadBatchCocoZip(imageName?: string): Promise<{ success: boolean; path?: string; error?: string }> {
  const query = imageName ? `?image_name=${encodeURIComponent(imageName)}` : "";
  const url = `${API_BASE}/batch/export/coco-zip${query}`;
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to export COCO dataset zip");
  }
  const blob = await res.blob();
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const defaultName = `reefvision_coco_dataset_${timestamp}.zip`;

  if (typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window)) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve) => {
        reader.onloadend = () => resolve(reader.result as string);
        reader.readAsDataURL(blob);
      });
      const dataUrl = await base64Promise;
      const saveRes = await invoke<any>("save_file_dialog", {
        defaultName,
        filterName: "ZIP Archive",
        extensions: ["zip"],
        content: dataUrl,
        isBase64: true,
      });
      return { success: saveRes.success, path: saveRes.path, error: saveRes.error };
    } catch (e) {
      console.warn("Tauri native save fallback to browser:", e);
    }
  }

  // Browser download fallback
  const a = document.createElement("a");
  const blobUrl = URL.createObjectURL(blob);
  a.href = blobUrl;
  a.download = defaultName;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
  }, 300);
  return { success: true, path: defaultName };
}

export async function getSegmentPolygons(
  imageName: string
): Promise<{ image_name: string; width: number; height: number; segments: CoralSegment[] }> {
  const res = await fetch(`${API_BASE}/analysis/polygons/${encodeURIComponent(imageName)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to load segment polygons");
  }
  return res.json();
}

export async function promptSegment(
  imageName: string,
  points: PromptPoint[],
  imageData?: string,
  signal?: AbortSignal
): Promise<PromptSegmentResponse> {
  const res = await fetch(`${API_BASE}/analysis/segment-prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      image_name: imageName,
      points,
      image_data: imageData && imageData.startsWith("data:") ? imageData : undefined,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Interactive segmentation failed");
  }
  return res.json();
}

export async function warmupSamPrompt(
  imageName: string,
  imageData?: string
): Promise<{ status: string; image_name?: string }> {
  try {
    const res = await fetch(`${API_BASE}/analysis/segment-prompt/warmup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_name: imageName,
        image_data: imageData && imageData.startsWith("data:") ? imageData : undefined,
      }),
    });
    if (!res.ok) return { status: "error" };
    return res.json();
  } catch {
    return { status: "error" };
  }
}

export async function saveAnnotations(
  imageName: string,
  segments: {
    id: number;
    genus: string;
    growth_form: string;
    condition: string;
    color_hex?: string;
    polygon: [number, number][];
  }[],
  imageData?: string
): Promise<SaveAnnotationsResponse> {
  const res = await fetch(`${API_BASE}/analysis/save-annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: imageName,
      segments,
      image_data: imageData && imageData.startsWith("data:") ? imageData : undefined,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to save annotations");
  }
  return res.json();
}

// --- Video Processing API Services ---

export async function uploadVideo(file: File): Promise<{
  status: string;
  video_name: string;
  video_path: string;
  metadata: import("../types").VideoMetadata;
}> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/video/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to upload video");
  }
  return res.json();
}

export async function processVideo(req: {
  video_name: string;
  overlap_threshold?: number;
  min_stride?: number;
  max_stride?: number;
  points_per_side?: number;
  classify?: boolean;
  export_mp4?: boolean;
}): Promise<{
  status: string;
  job_id: string;
  filename: string;
  metadata: import("../types").VideoMetadata;
}> {
  const res = await fetch(`${API_BASE}/video/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start video processing");
  }
  return res.json();
}

export async function getActiveVideoStatus(): Promise<import("../types").VideoJobStatus | null> {
  const res = await fetch(`${API_BASE}/video/active-status`);
  if (!res.ok) return null;
  return res.json();
}

export async function getVideoStatus(jobId: string): Promise<import("../types").VideoJobStatus> {
  const res = await fetch(`${API_BASE}/video/status/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to fetch video status");
  }
  return res.json();
}

export async function pauseVideoJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/video/pause/${jobId}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to pause video processing");
}

export async function resumeVideoJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/video/resume/${jobId}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to resume video processing");
}

export async function cancelVideoJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/video/cancel/${jobId}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to cancel video processing");
}

export async function getVideoFrameAnnotations(
  jobId: string,
  frameIdx: number
): Promise<{
  job_id: string;
  frame_idx: number;
  is_keyframe: boolean;
  tracks: import("../types").VideoTrackItem[];
  active_count: number;
}> {
  const res = await fetch(`${API_BASE}/video/frame/${jobId}/${frameIdx}`);
  if (!res.ok) throw new Error("Failed to fetch frame annotations");
  return res.json();
}

export async function getVideoFramesBulk(
  jobId: string,
  startFrame = 0,
  endFrame?: number
): Promise<{
  job_id: string;
  total_frames_available: number;
  frames: Record<string, { frame_idx: number; is_keyframe: boolean; tracks: import("../types").VideoTrackItem[] }>;
}> {
  const query = endFrame !== undefined ? `?start_frame=${startFrame}&end_frame=${endFrame}` : `?start_frame=${startFrame}`;
  const res = await fetch(`${API_BASE}/video/frames/${jobId}${query}`);
  if (!res.ok) throw new Error("Failed to fetch bulk frame annotations");
  return res.json();
}

export async function getVideoFramesDelta(
  jobId: string,
  sinceSeq = 0,
  limit = 100
): Promise<{
  job_id: string;
  since_seq: number;
  max_seq: number;
  count: number;
  frames: Array<{
    seq: number;
    frame_idx: number;
    timestamp_sec: number;
    is_keyframe: boolean;
    tracks: import("../types").VideoTrackItem[];
  }>;
}> {
  const res = await fetch(`${API_BASE}/video/delta/${jobId}?since_seq=${sinceSeq}&limit=${limit}`);
  if (!res.ok) throw new Error("Failed to fetch delta frame annotations");
  return res.json();
}

export function getVideoExportUrl(jobId: string, format: "mp4" | "csv" | "coco"): string {
  return `${API_BASE}/video/export/${jobId}?format=${format}`;
}

export async function downloadVideoExport(
  jobId: string,
  format: "mp4" | "csv" | "coco",
  defaultName: string,
  sourcePath?: string | null
): Promise<{ success: boolean; path?: string; cancelled?: boolean; error?: string }> {
  const isTauri = typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);

  // 1. If in Tauri desktop mode and we have the direct on-disk source path (e.g. for MP4)
  if (isTauri && sourcePath && format === "mp4") {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const res = await invoke<any>("copy_file_dialog", {
        defaultName,
        filterName: "MP4 Video",
        extensions: ["mp4"],
        sourcePath,
      });
      return { success: res.success, path: res.path, cancelled: res.cancelled, error: res.error };
    } catch (e: any) {
      console.warn("Tauri copy_file_dialog failed, falling back to HTTP fetch:", e);
    }
  }

  // 2. Fetch the file blob from backend
  const url = getVideoExportUrl(jobId, format);
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Export failed with status ${res.status}`);
  }
  const blob = await res.blob();

  // 3. If in Tauri and saving via save_file_dialog
  if (isTauri) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve) => {
        reader.onloadend = () => resolve(reader.result as string);
        reader.readAsDataURL(blob);
      });
      const dataUrl = await base64Promise;
      const filterName = format === "mp4" ? "MP4 Video" : format === "csv" ? "CSV Spreadsheet" : "COCO JSON";
      const ext = format === "coco" ? "json" : format;
      const saveRes = await invoke<any>("save_file_dialog", {
        defaultName,
        filterName,
        extensions: [ext],
        content: dataUrl,
        isBase64: true,
      });
      return { success: saveRes.success, path: saveRes.path, cancelled: saveRes.cancelled, error: saveRes.error };
    } catch (e: any) {
      console.warn("Tauri save_file_dialog failed, falling back to browser download:", e);
    }
  }

  // 4. Standard browser download fallback
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = defaultName;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
  }, 1000);

  return { success: true, path: defaultName };
}

export async function loadSampleVideo(): Promise<{
  status: string;
  video_name: string;
  video_path: string;
  metadata: import("../types").VideoMetadata;
}> {
  const res = await fetch(`${API_BASE}/video/sample`);
  if (!res.ok) throw new Error("Failed to load sample video");
  return res.json();
}

export function getVideoStreamUrl(videoName: string): string {
  return `${API_BASE}/video/stream-file/${encodeURIComponent(videoName)}`;
}




