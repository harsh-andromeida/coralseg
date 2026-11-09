import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  UploadCloud,
  FolderHeart,
  Rocket,
  DownloadCloud,
  AlertCircle,
  CheckCircle2,
  Check,
  Cpu,
  Zap,
  Trash2,
  X,
  Play,
  Pause,
  StopCircle,
  FolderArchive,
  ArrowRight,
  Loader2,
  Film,
  Video,
  Sparkles,
} from "lucide-react";
import JSZip from "jszip";
import { BrandLogo } from "../components/BrandLogo";
import { UISelect } from "../components/UISelect";
import { ExportDatasetModal } from "../components/ExportDatasetModal";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  DeviceInfo,
  ModelSpec,
  SampleItem,
  LoadedImage,
  UploadedVideoItem,
  DownloadProgress,
  BatchStatusResponse,
  VideoJobStatus,
} from "../types";
import {
  checkHealth,
  getModelsStatus,
  triggerModelDownload,
  getDownloadProgress,
  deleteModel,
  setDevicePreference,
  getSamples,
  getSampleImageUrl,
  registerImage,
  loadSampleToStore,
  startBatch,
  getBatchStatus,
  pauseBatch,
  resumeBatch,
  cancelBatch,
  uploadVideo,
  processVideo,
  getVideoStatus,
  getActiveVideoStatus,
  pauseVideoJob,
  resumeVideoJob,
  cancelVideoJob,
  loadSampleVideo,
  getVideoStreamUrl,
  getCustomApiUrl,
  setCustomApiUrl,
} from "../services/api";

interface ModelGroupDef {
  id: "sam" | "bioclip" | "bleaching";
  name: string;
  fullName: string;
  task: string;
  size: string;
  filenames: string[];
}

const MODEL_GROUPS: ModelGroupDef[] = [
  {
    id: "sam",
    name: "Segmentation Model",
    fullName: "SAM ViT-B Segmentation Model",
    task: "Dense Instance Segmentation",
    size: "~383 MB",
    filenames: ["sam_image_encoder.onnx", "sam_mask_decoder.onnx"],
  },
  {
    id: "bioclip",
    name: "Taxonomical Model",
    fullName: "BioCLIP Taxonomy Model",
    task: "Taxonomy & Growth Form",
    size: "~345 MB",
    filenames: ["bioclip_visual.onnx", "coral_taxonomy_embeddings.npy"],
  },
  {
    id: "bleaching",
    name: "Bleach Detection Model",
    fullName: "NOAA Bleach Detection Model",
    task: "Reef Health & Bleaching",
    size: "~6.1 MB",
    filenames: ["bleaching_yolo11n.onnx"],
  },
];

interface UploadViewProps {
  onLaunchStudio: (images: LoadedImage[]) => void;
  onLaunchVideoStudio?: (videoItem: UploadedVideoItem, jobId?: string) => void;
  activeVideo?: UploadedVideoItem | null;
  activeVideoJobId?: string;
  onSetActiveVideo?: (video: UploadedVideoItem | null) => void;
  onSetActiveVideoJobId?: (id: string) => void;
}

export const UploadView: React.FC<UploadViewProps> = ({
  onLaunchStudio,
  onLaunchVideoStudio,
  activeVideo,
  activeVideoJobId,
  onSetActiveVideo,
  onSetActiveVideoJobId,
}) => {
  const [activeTab, setActiveTab] = useState<"samples" | "upload" | "video">("samples");
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isCheckingBackend, setIsCheckingBackend] = useState<boolean>(true);
  const [allDownloaded, setAllDownloaded] = useState<boolean>(true);
  const [modelsList, setModelsList] = useState<ModelSpec[]>([]);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [devicePref, setDevicePref] = useState<string>("auto");
  const [concurrency, setConcurrency] = useState<number>(1);

  // Video State
  const [uploadedVideo, setUploadedVideo] = useState<UploadedVideoItem | null>(activeVideo || null);
  const [videoJobStatus, setVideoJobStatus] = useState<VideoJobStatus | null>(null);
  const [isPausingResumingVideo, setIsPausingResumingVideo] = useState<boolean>(false);
  const [videoDragActive, setVideoDragActive] = useState<boolean>(false);
  const [isUploadingVideo, setIsUploadingVideo] = useState<boolean>(false);
  const [isStartingVideoJob, setIsStartingVideoJob] = useState<boolean>(false);
  const [videoTrackerMode, setVideoTrackerMode] = useState<"sam2" | "optical_flow">("sam2");
  const [videoError, setVideoError] = useState<string | null>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  // Batch Background Processing State
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null);
  const [isStartingBatch, setIsStartingBatch] = useState<boolean>(false);
  const [isExportDatasetModalOpen, setIsExportDatasetModalOpen] = useState<boolean>(false);
  const [exportFeedback, setExportFeedback] = useState<string | null>(null);
  const [activeBatchImages, setActiveBatchImages] = useState<LoadedImage[]>([]);
  const [isCancellingBatch, setIsCancellingBatch] = useState<boolean>(false);

  // Download state
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress>({
    is_downloading: false,
    overall_pct: 0,
    current_file: "",
    detail: "",
  });
  const [isDeletingModel, setIsDeletingModel] = useState<string | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState<boolean>(false);

  // Staged upload images
  const [stagedImages, setStagedImages] = useState<LoadedImage[]>([]);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Samples
  const [availableSamples, setAvailableSamples] = useState<SampleItem[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>("First 6 Samples");
  const [selectedSpecific, setSelectedSpecific] = useState<string[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showRemoteModal, setShowRemoteModal] = useState<boolean>(false);
  const [remoteUrlInput, setRemoteUrlInput] = useState<string>(getCustomApiUrl());

  const handleSaveRemoteUrl = () => {
    setCustomApiUrl(remoteUrlInput);
    setShowRemoteModal(false);
    handleManualRetry();
  };

  const handleResetToLocal = () => {
    setCustomApiUrl(null);
    setRemoteUrlInput("");
    setShowRemoteModal(false);
    handleManualRetry();
  };

  // Poll backend health and initialize with continuous heartbeat
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let isSubscribed = true;

    const pingAndSync = async () => {
      try {
        const health = await checkHealth();
        if (!isSubscribed) return;
        setIsBackendConnected(true);
        setIsCheckingBackend(false);
        setDeviceInfo(health.device);
        if (health.device?.concurrency) {
          setConcurrency(health.device.concurrency);
        }

        setAvailableSamples((prev) => {
          if (prev.length === 0) {
            getSamples()
              .then((samplesRes) => {
                if (isSubscribed) {
                  setAvailableSamples(samplesRes);
                  if (samplesRes.length > 0) {
                    setSelectedSpecific(samplesRes.slice(0, 4).map((s) => s.filename));
                  }
                }
              })
              .catch(console.error);

            getModelsStatus()
              .then((modelsRes) => {
                if (isSubscribed) {
                  setAllDownloaded(modelsRes.all_downloaded);
                  setModelsList(modelsRes.models);
                  setDeviceInfo(modelsRes.device_info);
                  if (modelsRes.device_info?.concurrency) {
                    setConcurrency(modelsRes.device_info.concurrency);
                  }
                }
              })
              .catch(console.error);
          }
          return prev;
        });

        // Heartbeat check every 3s
        timer = setTimeout(pingAndSync, 3000);
      } catch (err) {
        if (!isSubscribed) return;
        setIsBackendConnected(false);
        setIsCheckingBackend(false);
        // Retry polling every 1.5 seconds until backend is online
        timer = setTimeout(pingAndSync, 1500);
      }
    };

    pingAndSync();

    return () => {
      isSubscribed = false;
      clearTimeout(timer);
    };
  }, []);

  // Continuous background batch status poller
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let isSubscribed = true;

    const pollBatch = async () => {
      try {
        const res = await getBatchStatus();
        if (isSubscribed) {
          setBatchStatus(res);
        }
      } catch {
        // silent if backend offline
      }
      if (isSubscribed) {
        timer = setTimeout(pollBatch, 1000);
      }
    };

    pollBatch();

    return () => {
      isSubscribed = false;
      clearTimeout(timer);
    };
  }, []);

  // Continuous background video transect status poller
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let isSubscribed = true;

    const pollVideo = async () => {
      try {
        const targetId = activeVideoJobId || videoJobStatus?.job_id;
        let s: VideoJobStatus | null = null;
        if (targetId) {
          s = await getVideoStatus(targetId);
        } else {
          s = await getActiveVideoStatus();
        }
        if (isSubscribed && s) {
          setVideoJobStatus(s);
          if (!uploadedVideo && s.filename) {
            const previewUrl = getVideoStreamUrl(s.filename);
            const recovered: UploadedVideoItem = {
              file: new File([], s.filename),
              video_name: s.filename,
              video_path: s.metadata?.filename,
              metadata: s.metadata,
              previewUrl,
              job_id: s.job_id,
            };
            setUploadedVideo(recovered);
            onSetActiveVideo?.(recovered);
            onSetActiveVideoJobId?.(s.job_id);
          }
        }
      } catch {
        // silent poll
      }
      if (isSubscribed) {
        timer = setTimeout(pollVideo, 1000);
      }
    };

    pollVideo();

    return () => {
      isSubscribed = false;
      clearTimeout(timer);
    };
  }, [activeVideoJobId, videoJobStatus?.job_id, uploadedVideo]);

  const handleManualRetry = async () => {
    setIsCheckingBackend(true);
    setLoadError(null);
    try {
      const health = await checkHealth();
      setIsBackendConnected(true);
      setDeviceInfo(health.device);
      if (health.device?.concurrency) {
        setConcurrency(health.device.concurrency);
      }

      const [modelsRes, samplesRes] = await Promise.all([
        getModelsStatus(),
        getSamples(),
      ]);
      setAllDownloaded(modelsRes.all_downloaded);
      setModelsList(modelsRes.models);
      setDeviceInfo(modelsRes.device_info);
      if (modelsRes.device_info?.concurrency) {
        setConcurrency(modelsRes.device_info.concurrency);
      }
      setAvailableSamples(samplesRes);
      if (samplesRes.length > 0) {
        setSelectedSpecific(samplesRes.slice(0, 4).map((s) => s.filename));
      }
    } catch (e) {
      setIsBackendConnected(false);
      setLoadError("Could not connect to Python backend at http://127.0.0.1:8000");
    } finally {
      setIsCheckingBackend(false);
    }
  };

  // Device preference change
  const handleDeviceChange = async (val: string) => {
    setDevicePref(val);
    try {
      const updated = await setDevicePreference(val, concurrency);
      setDeviceInfo(updated);
    } catch (e) {
      console.error("Failed to update device", e);
    }
  };

  const handleConcurrencyChange = async (c: number) => {
    setConcurrency(c);
    try {
      const updated = await setDevicePreference(devicePref, c);
      setDeviceInfo(updated);
    } catch (e) {
      console.error("Failed to update concurrency", e);
    }
  };

  // Trigger Download & Poll Progress (Re-download removed)
  const handleStartDownload = async (target?: string | string[]) => {
    try {
      await triggerModelDownload(false, target);
      const interval = setInterval(async () => {
        try {
          const prog = await getDownloadProgress();
          setDownloadProgress(prog);
          if (!prog.is_downloading && prog.overall_pct === 100) {
            clearInterval(interval);
            const modelsRes = await getModelsStatus();
            setAllDownloaded(modelsRes.all_downloaded);
            setModelsList(modelsRes.models);
          } else if (prog.error) {
            clearInterval(interval);
          }
        } catch {
          clearInterval(interval);
        }
      }, 500);
    } catch (e) {
      alert(`Download trigger error: ${e}`);
    }
  };

  // Delete one model group or specific file
  const handleDeleteModel = async (target: string) => {
    setIsDeletingModel(target);
    setLoadError(null);
    try {
      const res = await deleteModel(target);
      setAllDownloaded(res.all_downloaded);
      setModelsList(res.models);
    } catch (e: any) {
      alert(`Failed to delete model: ${e?.message || e}`);
    } finally {
      setIsDeletingModel(null);
    }
  };

  // Check readiness of model groups
  const isGroupReady = (groupId: string): boolean => {
    if (allDownloaded) return true;
    const group = MODEL_GROUPS.find((g) => g.id === groupId);
    if (!group) return false;
    return group.filenames.every((fname) => {
      const item = modelsList.find((m) => m.filename === fname || m.filename.includes(fname));
      return item?.cached ?? false;
    });
  };

  const getEngineName = (info: DeviceInfo | null): string => {
    if (!info) return "Detecting compute hardware...";
    if (info.active_provider?.includes("CUDA")) {
      return "NVIDIA CUDA GPU";
    }
    if (info.cuda_status === "onnxruntime_gpu_missing") {
      return "CPU (GPU Needs onnxruntime-gpu)";
    }
    if (info.active_provider?.includes("CoreML") || info.gpu_name?.includes("MPS")) {
      return "Apple Silicon (MPS)";
    }
    if (info.mode === "cpu") {
      return "Single-Threaded CPU Engine (1 Thread)";
    }
    if (info.cpu_count && info.cpu_count > 1) {
      return `Multi-Threaded CPU Engine (${info.cpu_count} Threads)`;
    }
    return info.gpu_name || "CPU Engine";
  };

  // Compute active sample subset
  const sampleSubset = useMemo(() => {
    if (selectedPreset === "First 6 Samples") return availableSamples.slice(0, 6);
    if (selectedPreset === "First 12 Samples") return availableSamples.slice(0, 12);
    if (selectedPreset === "First 24 Samples") return availableSamples.slice(0, 24);
    if (selectedPreset === "All Available Samples") return availableSamples;
    return availableSamples.filter((s) => selectedSpecific.includes(s.filename));
  }, [selectedPreset, availableSamples, selectedSpecific]);

  // File Upload Handlers
  const handleFiles = async (fileList: FileList) => {
    setLoadError(null);
    const newImages: LoadedImage[] = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const name = file.name;
      const lower = name.toLowerCase();

      if (lower.endsWith(".zip")) {
        try {
          const zip = await JSZip.loadAsync(file);
          const entries = Object.keys(zip.files);
          for (const filename of entries) {
            const entry = zip.files[filename];
            if (entry.dir) continue;
            const baseName = filename.split("/").pop() || "";
            if (baseName.startsWith(".") || filename.includes("__MACOSX")) continue;

            const ext = baseName.split(".").pop()?.toLowerCase() || "";
            if (["jpg", "jpeg", "png", "webp"].includes(ext)) {
              const blob = await entry.async("blob");
              const dataUrl = await new Promise<string>((resolve) => {
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result as string);
                reader.readAsDataURL(blob);
              });
              newImages.push({ name: baseName, dataUrl });
            }
          }
        } catch (err) {
          console.error(`Error reading zip ${name}:`, err);
        }
      } else {
        const ext = name.split(".").pop()?.toLowerCase() || "";
        if (["jpg", "jpeg", "png", "webp"].includes(ext)) {
          const dataUrl = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result as string);
            reader.readAsDataURL(file);
          });
          newImages.push({ name, dataUrl });
        }
      }
    }

    setStagedImages((prev) => [...prev, ...newImages]);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const onDragLeave = () => setDragActive(false);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  // Start Batch for Staged Upload Images
  const handleStartBatchUpload = async () => {
    if (stagedImages.length === 0) {
      alert("Please upload at least 1 image first.");
      return;
    }
    const modelsRes = await getModelsStatus().catch(() => null);
    if (!modelsRes || !modelsRes.all_downloaded) {
      setAllDownloaded(false);
      if (modelsRes) setModelsList(modelsRes.models);
      setLoadError("Process blocked: Foundation models are missing or deleted. You must download all models before proceeding.");
      return;
    }
    setIsStartingBatch(true);
    setLoadError(null);
    try {
      for (const img of stagedImages) {
        await registerImage(img.name, img.dataUrl);
      }
      setActiveBatchImages(stagedImages);
      const res = await startBatch(stagedImages.map((img) => img.name), undefined, concurrency);
      setBatchStatus(res);
    } catch (e: any) {
      setLoadError(`Failed to start batch processing: ${e?.message || e}`);
    } finally {
      setIsStartingBatch(false);
    }
  };

  // Start Batch for Sample Library Images
  const handleStartBatchSamples = async () => {
    if (sampleSubset.length === 0) {
      alert("No sample images available to launch. Please select at least 1 file.");
      return;
    }
    const modelsRes = await getModelsStatus().catch(() => null);
    if (!modelsRes || !modelsRes.all_downloaded) {
      setAllDownloaded(false);
      if (modelsRes) setModelsList(modelsRes.models);
      setLoadError("Process blocked: Foundation models are missing or deleted. You must download all models before proceeding.");
      return;
    }
    setIsStartingBatch(true);
    setLoadError(null);
    try {
      await checkHealth();
      const loaded: LoadedImage[] = [];
      for (const s of sampleSubset) {
        await loadSampleToStore(s.filename);
        loaded.push({
          name: s.filename,
          dataUrl: getSampleImageUrl(s.filename),
        });
      }
      setActiveBatchImages(loaded);
      const res = await startBatch(sampleSubset.map((s) => s.filename), undefined, concurrency);
      setBatchStatus(res);
    } catch (e: any) {
      setIsBackendConnected(false);
      setLoadError(`Vision engine at http://127.0.0.1:8000 is unreachable: ${e?.message || e}`);
    } finally {
      setIsStartingBatch(false);
    }
  };

  const handlePauseResumeBatch = async () => {
    try {
      if (batchStatus?.is_paused) {
        const res = await resumeBatch();
        setBatchStatus(res);
      } else {
        const res = await pauseBatch();
        setBatchStatus(res);
      }
    } catch (e: any) {
      alert(`Error toggling pause: ${e?.message || e}`);
    }
  };

  const handleCancelBatchRun = async () => {
    setIsCancellingBatch(true);
    try {
      const res = await cancelBatch();
      setBatchStatus(res);
      setExportFeedback("Processing cancelled safely. Models unloaded from RAM.");
      setTimeout(() => setExportFeedback(null), 5000);
    } catch (e: any) {
      alert(`Error cancelling batch: ${e?.message || e}`);
    } finally {
      setIsCancellingBatch(false);
    }
  };

  // Video Handlers
  const handleVideoFileSelect = async (file: File) => {
    setIsUploadingVideo(true);
    setVideoError(null);
    try {
      const uploadRes = await uploadVideo(file);
      const previewUrl = URL.createObjectURL(file);
      setUploadedVideo({
        file,
        video_name: uploadRes.video_name,
        video_path: uploadRes.video_path,
        metadata: uploadRes.metadata,
        previewUrl,
      });
    } catch (err: any) {
      setVideoError(`Video upload failed: ${err?.message || err}`);
    } finally {
      setIsUploadingVideo(false);
    }
  };

  const handleLoadSampleVideo = async () => {
    setIsUploadingVideo(true);
    setVideoError(null);
    try {
      const sampleRes = await loadSampleVideo();
      const previewUrl = getVideoStreamUrl(sampleRes.video_name);
      setUploadedVideo({
        file: new File([], sampleRes.video_name),
        video_name: sampleRes.video_name,
        video_path: sampleRes.video_path,
        metadata: sampleRes.metadata,
        previewUrl,
      });
    } catch (err: any) {
      setVideoError(`Failed to load sample video: ${err?.message || err}`);
    } finally {
      setIsUploadingVideo(false);
    }
  };

  const handleStartVideoAnalysis = async () => {
    if (!uploadedVideo) return;
    setIsStartingVideoJob(true);
    setVideoError(null);

    const cfg = {
      max_stride: 25,
      min_stride: 10,
      points_per_side: 16,
      overlap_threshold: 0.70,
    };

    try {
      const procRes = await processVideo({
        video_name: uploadedVideo.video_name,
        overlap_threshold: cfg.overlap_threshold,
        min_stride: cfg.min_stride,
        max_stride: cfg.max_stride,
        points_per_side: cfg.points_per_side,
        tracker_mode: videoTrackerMode,
      });

      onSetActiveVideo?.(uploadedVideo);
      onSetActiveVideoJobId?.(procRes.job_id);

      if (onLaunchVideoStudio) {
        onLaunchVideoStudio(uploadedVideo, procRes.job_id);
      }
    } catch (err: any) {
      setVideoError(`Failed to start video processing: ${err?.message || err}`);
    } finally {
      setIsStartingVideoJob(false);
    }
  };

  const handlePauseResumeVideo = async () => {
    if (!videoJobStatus) return;
    setIsPausingResumingVideo(true);
    try {
      if (videoJobStatus.state === "paused") {
        await resumeVideoJob(videoJobStatus.job_id);
      } else {
        await pauseVideoJob(videoJobStatus.job_id);
      }
      const updated = await getVideoStatus(videoJobStatus.job_id);
      setVideoJobStatus(updated);
    } catch (err: any) {
      alert(`Error toggling video pause: ${err?.message || err}`);
    } finally {
      setIsPausingResumingVideo(false);
    }
  };

  const handleCancelVideo = async () => {
    if (!videoJobStatus) return;
    try {
      await cancelVideoJob(videoJobStatus.job_id);
      const updated = await getVideoStatus(videoJobStatus.job_id);
      setVideoJobStatus(updated);
    } catch (err: any) {
      alert(`Error cancelling video: ${err?.message || err}`);
    }
  };

  const handleOpenVideoStudioFromPanel = () => {
    if (!videoJobStatus) return;
    const videoItem = uploadedVideo || {
      file: new File([], videoJobStatus.filename),
      video_name: videoJobStatus.filename,
      video_path: videoJobStatus.metadata?.filename,
      metadata: videoJobStatus.metadata,
      previewUrl: getVideoStreamUrl(videoJobStatus.filename),
      job_id: videoJobStatus.job_id,
    };
    onLaunchVideoStudio?.(videoItem, videoJobStatus.job_id);
  };

  const handleLaunchStudioNow = () => {
    if (activeBatchImages.length > 0) {
      onLaunchStudio(activeBatchImages);
    } else if (activeTab === "upload" && stagedImages.length > 0) {
      onLaunchStudio(stagedImages);
    } else {
      const loaded = sampleSubset.map((s) => ({
        name: s.filename,
        dataUrl: getSampleImageUrl(s.filename),
      }));
      onLaunchStudio(loaded);
    }
  };

  // Reusable video transect control & progress panel
  const renderVideoControlPanel = () => {
    const isRunning = videoJobStatus?.state === "processing";
    const isPaused = videoJobStatus?.state === "paused";
    const isComplete = videoJobStatus?.state === "completed";
    const hasProcessed = (videoJobStatus?.current_frame ?? 0) > 0;

    if (!isRunning && !isPaused && !isComplete && !hasProcessed) {
      return (
        <div className="shrink-0 pt-2.5 mt-auto border-t border-slate-100 bg-white">
          <Button
            type="button"
            className="w-full h-10 text-xs font-bold tracking-wider uppercase text-white rounded-xl shadow-xs hover:shadow-md transition-all gap-2 bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] hover:opacity-95"
            onClick={handleStartVideoAnalysis}
            disabled={!uploadedVideo || isStartingVideoJob || isUploadingVideo || !isBackendConnected}
          >
            <Rocket size={14} className={isStartingVideoJob ? "animate-bounce" : ""} />
            <span>
              {isUploadingVideo
                ? "Uploading Video..."
                : isStartingVideoJob
                ? "Starting Video Tracker..."
                : uploadedVideo
                ? "Launch Video Tracking & Transect Studio"
                : "Select or Drag Video Above to Begin"}
            </span>
          </Button>
        </div>
      );
    }

    return (
      <div className="shrink-0 pt-2.5 mt-auto border-t border-slate-200/90 bg-white">
        <div className="rounded-xl border border-cyan-200 bg-cyan-50/70 p-3 shadow-xs space-y-2.5">
          {/* Row 1: Header status and controls */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              {isRunning && (
                <span className="relative flex h-2.5 w-2.5 shrink-0">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-600" />
                </span>
              )}
              {isPaused && <span className="h-2.5 w-2.5 rounded-full bg-amber-500 shrink-0" />}
              {isComplete && <CheckCircle2 size={15} className="text-emerald-600 shrink-0" />}

              <span className="text-xs font-bold text-slate-800 truncate">
                {isComplete
                  ? "Transect Survey Complete"
                  : isPaused
                  ? "Transect Tracking Paused"
                  : "Autonomous Transect Processing"}
              </span>

              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border border-cyan-200 text-cyan-800 font-bold shrink-0">
                {videoJobStatus?.current_frame || 0} / {videoJobStatus?.total_frames || 0} Frames
              </span>

              {videoJobStatus?.fps_realtime ? (
                <span className="text-[9px] font-mono text-cyan-700 bg-cyan-100/80 px-1 rounded shrink-0">
                  {videoJobStatus.fps_realtime} FPS
                </span>
              ) : null}
            </div>

            <div className="flex items-center gap-1.5 shrink-0">
              {(isRunning || isPaused) && (
                <>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handlePauseResumeVideo}
                    disabled={isPausingResumingVideo}
                    className="h-7 px-2 text-[10px] font-semibold gap-1 text-slate-700 hover:bg-slate-100"
                    title={isPaused ? "Resume processing" : "Pause processing"}
                  >
                    {isPaused ? <Play size={11} /> : <Pause size={11} />}
                    <span>{isPaused ? "Resume" : "Pause"}</span>
                  </Button>

                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleCancelVideo}
                    className="h-7 px-2 text-[10px] font-semibold gap-1 text-red-600 border-red-200 hover:bg-red-50"
                    title="Cancel video tracking"
                  >
                    <StopCircle size={11} />
                    <span>Cancel</span>
                  </Button>
                </>
              )}

              {isComplete && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleStartVideoAnalysis}
                  disabled={isStartingVideoJob}
                  className="h-7 px-2 text-[10px] font-semibold gap-1 text-slate-600 hover:bg-slate-100"
                  title="Re-run tracking from start"
                >
                  <Rocket size={11} />
                  <span>Re-run</span>
                </Button>
              )}
            </div>
          </div>

          {/* Row 2: Active Stage & Model Info */}
          <div className="flex items-center justify-between text-[11px] text-slate-600 gap-2">
            <div className="flex items-center gap-1.5 truncate">
              {videoJobStatus?.current_model && (
                <span className="text-[10px] bg-white border border-cyan-200 text-cyan-800 font-semibold px-1.5 py-0.2 rounded shrink-0">
                  {videoJobStatus.current_model}
                </span>
              )}
              <span className="truncate">
                {videoJobStatus?.current_stage || "Processing seabed video transect..."}
              </span>
            </div>

            <span className="font-mono font-bold text-cyan-700 shrink-0">
              {videoJobStatus?.progress_pct || 0}%
            </span>
          </div>

          {/* Row 3: Progress bar */}
          <Progress
            value={videoJobStatus?.progress_pct || 0}
            className="h-2 bg-cyan-100"
            indicatorColor={isRunning ? "bg-gradient-to-r from-cyan-600 to-teal-500" : isPaused ? "bg-amber-500" : "bg-emerald-600"}
          />

          {/* Row 4: Colonies count and Open Studio Action */}
          <div className="flex items-center justify-between pt-1 border-t border-cyan-100/80">
            <div className="text-[10px] text-slate-500 flex items-center gap-2">
              <span>{videoJobStatus?.total_unique_corals || 0} Unique Corals</span>
              <span>•</span>
              <span>{videoJobStatus?.keyframes_count || 0} Keyframes</span>
              <span>•</span>
              <span>{videoJobStatus?.bleaching_prevalence_pct || 0}% Bleached</span>
            </div>

            <Button
              type="button"
              size="sm"
              onClick={handleOpenVideoStudioFromPanel}
              className="h-7 px-3 text-xs font-bold gap-1.5 bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] text-white hover:opacity-95 shadow-xs"
            >
              <span>Open Video Studio ({videoJobStatus?.current_frame || 0} Frames Ready)</span>
              <ArrowRight size={12} />
            </Button>
          </div>
        </div>
      </div>
    );
  };

  // Reusable batch control & progress panel
  const renderBatchControlPanel = (itemsCount: number, onStartBatch: () => void, isStarting: boolean) => {
    const isRunning = batchStatus?.is_running ?? false;
    const isPaused = batchStatus?.is_paused ?? false;
    const isComplete = !isRunning && (batchStatus?.completed ?? 0) > 0 && (batchStatus?.completed === batchStatus?.total);
    const hasProcessed = (batchStatus?.completed ?? 0) > 0;

    if (!isRunning && !isPaused && !isComplete && !hasProcessed) {
      return (
        <div className="shrink-0 pt-2.5 mt-auto border-t border-slate-100 bg-white">
          <Button
            type="button"
            className="w-full h-10 text-xs font-bold tracking-wider uppercase text-white rounded-xl shadow-xs hover:shadow-md transition-all gap-2 bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] hover:opacity-95 active:scale-[0.99]"
            onClick={onStartBatch}
            disabled={isStarting || itemsCount === 0 || !isBackendConnected}
          >
            <Rocket size={14} className={isStarting ? "animate-bounce" : ""} />
            <span>
              {isStarting
                ? "Initializing Background Engines..."
                : !isBackendConnected
                ? "Waiting for Vision Engine..."
                : `Start Batch Processing (${itemsCount} Images)`}
            </span>
          </Button>
        </div>
      );
    }

    return (
      <div className="shrink-0 pt-2.5 mt-auto border-t border-slate-200/90 bg-white">
        {exportFeedback && (
          <div className="mb-2 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between">
            <span>{exportFeedback}</span>
            <button onClick={() => setExportFeedback(null)} className="text-emerald-600 hover:text-emerald-900">
              <X size={12} />
            </button>
          </div>
        )}

        <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3 shadow-xs space-y-2.5">
          {/* Row 1: Header status and controls */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              {isRunning && !isPaused && (
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-teal-600" />
                </span>
              )}
              {isPaused && <span className="h-2.5 w-2.5 rounded-full bg-amber-500" />}
              {isComplete && <CheckCircle2 size={14} className="text-emerald-600 shrink-0" />}

              <span className="text-xs font-bold text-slate-800 truncate">
                {isComplete
                  ? "Batch Complete"
                  : isPaused
                  ? "Batch Paused"
                  : isRunning
                  ? "Autonomous Background Processing"
                  : "Batch Idle"}
              </span>

              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-600">
                {batchStatus?.completed || 0} / {batchStatus?.total || itemsCount} Done
              </span>
            </div>

            <div className="flex items-center gap-1.5 shrink-0">
              {(isRunning || isPaused) && (
                <>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handlePauseResumeBatch}
                    disabled={isCancellingBatch}
                    className="h-7 px-2 text-[10px] font-semibold gap-1 text-slate-700 hover:bg-slate-100"
                    title={isPaused ? "Resume processing" : "Pause processing"}
                  >
                    {isPaused ? <Play size={11} /> : <Pause size={11} />}
                    <span>{isPaused ? "Resume" : "Pause"}</span>
                  </Button>

                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleCancelBatchRun}
                    disabled={isCancellingBatch}
                    className="h-7 px-2 text-[10px] font-semibold gap-1 text-red-600 hover:bg-red-50 hover:border-red-200 border-red-200"
                    title="Cancel processing and unload models from RAM"
                  >
                    {isCancellingBatch ? (
                      <Loader2 size={11} className="animate-spin text-red-600" />
                    ) : (
                      <StopCircle size={11} />
                    )}
                    <span>{isCancellingBatch ? "Unloading..." : "Cancel"}</span>
                  </Button>
                </>
              )}

              {!isRunning && !isPaused && hasProcessed && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={onStartBatch}
                  className="h-7 px-2 text-[10px] font-semibold gap-1 text-teal-700 hover:bg-teal-50"
                  title="Re-run batch on selected images"
                >
                  <Rocket size={11} />
                  <span>Restart</span>
                </Button>
              )}
            </div>
          </div>

          {/* Row 2: Progress bar & Live stage readout with RAM model loading indicator */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-[11px] font-medium text-slate-600">
              <div className="flex items-center gap-1.5 min-w-0 pr-2">
                {batchStatus?.current_stage?.includes("Loading") && (
                  <span className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-900 border border-amber-300 shrink-0 animate-pulse">
                    <Cpu size={10} className="text-amber-700" />
                    <span>Loading Model into RAM</span>
                  </span>
                )}
                <span className="truncate max-w-[340px]">
                  {isRunning && !isPaused
                    ? batchStatus?.current_stage?.includes("Loading")
                      ? batchStatus.current_stage
                      : batchStatus?.current_image
                      ? `Analyzing ${batchStatus.current_image} • ${batchStatus.current_stage || "Processing"}`
                      : batchStatus?.current_stage || "Processing..."
                    : isPaused
                    ? "Paused (Current progress preserved)"
                    : batchStatus?.current_stage?.includes("Cancelled") || batchStatus?.current_stage === "cancelled"
                    ? "Cancelled • Models unloaded from RAM"
                    : isComplete
                    ? "All images segmented and taxonomically enriched"
                    : "Ready for analysis"}
                </span>
              </div>
              <span className="font-mono text-teal-700 font-bold shrink-0">
                {batchStatus?.percent || 0}%
              </span>
            </div>
            <Progress
              value={batchStatus?.percent || 0}
              className="h-2 bg-slate-200/80 rounded-full"
              indicatorColor={
                batchStatus?.current_stage?.includes("Loading")
                  ? "bg-gradient-to-r from-amber-500 via-orange-500 to-amber-600 animate-pulse"
                  : "bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85]"
              }
            />
          </div>

          {/* Row 3: Action Buttons (View Progress & Download COCO Dataset) */}
          <div className="flex items-center gap-2 pt-1">
            <Button
              type="button"
              className="flex-1 h-9 text-xs font-bold uppercase tracking-wider text-white rounded-lg shadow-xs hover:shadow-md transition-all gap-1.5 bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] hover:opacity-95"
              onClick={handleLaunchStudioNow}
              disabled={!isBackendConnected || (!hasProcessed && !isRunning)}
            >
              <span>
                {hasProcessed
                  ? `Open Studio (${batchStatus?.completed} Ready)`
                  : "View Progress in Studio"}
              </span>
              <ArrowRight size={13} />
            </Button>

            <Button
              type="button"
              variant="outline"
              className="h-9 px-3 text-xs font-bold gap-1.5 border-teal-300 text-teal-800 bg-teal-50/60 hover:bg-teal-100 hover:border-teal-400 rounded-lg transition-colors cursor-pointer"
              onClick={() => setIsExportDatasetModalOpen(true)}
              disabled={!hasProcessed}
              title="Export complete structured dataset into a custom destination folder or as a .zip"
            >
              <FolderArchive size={14} className="text-teal-700" />
              <span>Export Dataset</span>
            </Button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="portal-container">
      <div className="portal-card">
        {/* Left Column: System Status & Hardware Acceleration */}
        <div className="portal-brand-col">
          <BrandLogo />

          {/* System & Model Status Panel */}
          <div className="rounded-xl border border-slate-200/90 bg-slate-50/50 p-2.5 shadow-xs mb-2.5">
            {/* Row 1: Vision Engine Connected */}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                {isBackendConnected ? (
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                  </span>
                ) : (
                  <span className={`h-2 w-2 rounded-full ${isCheckingBackend ? "bg-amber-400 animate-pulse" : "bg-red-500"}`} />
                )}
                <span className="text-xs font-bold text-slate-800">Vision Engine</span>
              </div>

              {isBackendConnected ? (
                <div className="flex items-center gap-1.5">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/70">
                    <Check size={10} strokeWidth={2.5} />
                    Connected
                  </span>
                  {getCustomApiUrl() && (
                    <button
                      type="button"
                      onClick={() => setShowRemoteModal(true)}
                      className="text-[9.5px] font-medium text-teal-700 hover:underline px-1 py-0.5 bg-teal-50 rounded border border-teal-200/60"
                      title={`Connected to remote backend: ${getCustomApiUrl()}`}
                    >
                      Remote
                    </button>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] font-medium text-amber-700">
                    {isCheckingBackend ? "Connecting..." : "Offline"}
                  </span>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-6 px-1.5 text-[10px] font-semibold border-amber-300 hover:bg-amber-50"
                    onClick={handleManualRetry}
                  >
                    Retry
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-6 px-1.5 text-[10px] font-semibold text-teal-700 hover:bg-teal-50"
                    onClick={() => setShowRemoteModal(true)}
                    title="Connect to Google Colab or remote tunnel"
                  >
                    Remote/Colab
                  </Button>
                </div>
              )}
            </div>

            {isBackendConnected && (
              <>
                <div className="h-px bg-slate-200/70 my-2" />

                {/* Row 2: Foundation Models header + Delete/Manage Button */}
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <div className="flex items-center gap-1.5">
                    {allDownloaded ? (
                      <CheckCircle2 size={13} className="text-emerald-600" />
                    ) : (
                      <AlertCircle size={13} className="text-amber-600" />
                    )}
                    <span className="text-xs font-bold text-slate-800">Foundation Models</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-md border ${
                        allDownloaded
                          ? "text-emerald-700 bg-emerald-50 border-emerald-200/70"
                          : "text-amber-700 bg-amber-50 border-amber-200/70"
                      }`}
                    >
                      {allDownloaded ? "Ready" : "Download Required"}
                    </span>
                    {allDownloaded && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-6 px-2 text-[10px] font-semibold text-slate-500 hover:text-red-600 hover:bg-red-50 hover:border-red-200 border-slate-200 gap-1 rounded-md transition-colors"
                        onClick={() => setShowDeleteModal(true)}
                        disabled={downloadProgress.is_downloading || !!isDeletingModel}
                        title="Manage and delete individual foundation models"
                      >
                        <Trash2 className="h-3 w-3" />
                        <span>Manage</span>
                      </Button>
                    )}
                  </div>
                </div>

                {/* Row 3: Models list with exact names requested by user */}
                <div className="flex flex-col gap-1">
                  {MODEL_GROUPS.map((group) => {
                    const ready = isGroupReady(group.id);
                    return (
                      <div
                        key={group.id}
                        className="flex items-center justify-between px-2 py-1 rounded-lg bg-white border border-slate-200/70 text-xs hover:border-slate-300 transition-colors shadow-2xs"
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          {ready ? (
                            <Check size={12} className="text-emerald-600 shrink-0" strokeWidth={2.5} />
                          ) : (
                            <AlertCircle size={12} className="text-amber-500 shrink-0" />
                          )}
                          <span className="font-semibold text-slate-700 text-[11px] truncate" title={group.fullName}>
                            {group.name}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0 ml-2">
                          <span className="text-[10px] font-mono text-slate-400 bg-slate-50 border border-slate-200/60 px-1.5 py-0.5 rounded">
                            {group.size}
                          </span>
                          {ready && (
                            <button
                              type="button"
                              className="text-slate-300 hover:text-red-500 transition-colors p-0.5 rounded"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteModel(group.id);
                              }}
                              disabled={downloadProgress.is_downloading || !!isDeletingModel}
                              title={`Delete ${group.name} (${group.size})`}
                            >
                              <Trash2
                                size={11}
                                className={isDeletingModel === group.id ? "animate-spin text-red-500" : ""}
                              />
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {downloadProgress.is_downloading && (
                  <div className="mt-2.5 p-2.5 rounded-lg bg-white border border-slate-200/80">
                    <div className="flex justify-between items-center text-[11px] font-semibold text-slate-800 mb-1">
                      <span className="truncate max-w-[200px]">{downloadProgress.current_file}</span>
                      <span className="text-teal-700 font-mono">{downloadProgress.overall_pct}%</span>
                    </div>
                    <Progress
                      value={downloadProgress.overall_pct}
                      className="h-1.5 bg-slate-100 my-1"
                      indicatorColor="bg-[#0d7c85]"
                    />
                    <div className="text-[10px] text-slate-500">{downloadProgress.detail}</div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Hardware Acceleration Subcard */}
          <div className="rounded-xl border border-slate-200/90 bg-white p-2.5 shadow-xs">
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-1.5">
                <Cpu size={14} className="text-teal-600" />
                <span className="text-[11px] font-bold text-[#0f1e4a] tracking-wider uppercase">
                  Hardware Acceleration
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                <span className="text-[10px] font-semibold text-emerald-700 uppercase tracking-wide">
                  Active
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between gap-2 p-1.5 rounded-lg bg-slate-50 border border-slate-100 mb-2">
              <span className="text-[11px] font-medium text-slate-500">Active Engine</span>
              <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-semibold bg-white border border-slate-200 text-slate-700 shadow-2xs">
                {deviceInfo?.active_provider?.includes("CUDA") ? (
                  <Zap size={11} className="text-amber-500" />
                ) : deviceInfo?.cuda_status === "onnxruntime_gpu_missing" ? (
                  <Zap size={11} className="text-amber-500 animate-pulse" />
                ) : (
                  <Cpu size={11} className="text-teal-600" />
                )}
                <span className="truncate max-w-[170px]">{getEngineName(deviceInfo)}</span>
              </div>
            </div>

            {deviceInfo?.cuda_status === "onnxruntime_gpu_missing" && (
              <div className="p-2 mb-2 rounded-lg bg-amber-50/90 border border-amber-200 text-amber-900 text-[11px] flex flex-col gap-1.5 shadow-2xs">
                <div className="flex items-center gap-1.5 font-semibold text-amber-950">
                  <Zap size={12} className="text-amber-600 fill-amber-500" />
                  <span>NVIDIA GPU Detected (CPU Fallback)</span>
                </div>
                <p className="text-[10px] text-amber-800 leading-tight">
                  PyTorch detects CUDA, but ONNX is running on CPU because <code className="bg-amber-100/80 px-1 py-0.5 rounded font-mono">onnxruntime-gpu</code> is not installed. To unlock 50x–100x GPU acceleration:
                </p>
                <div className="flex items-center justify-between gap-1 bg-white p-1.5 rounded border border-amber-200">
                  <code className="text-[9px] font-mono text-slate-700 select-all truncate">
                    pip uninstall -y onnxruntime && pip install onnxruntime-gpu
                  </code>
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard.writeText("pip uninstall -y onnxruntime && pip install onnxruntime-gpu");
                      alert("Command copied to clipboard! Run it in your Python / Colab environment.");
                    }}
                    className="px-2 py-0.5 text-[9px] font-semibold bg-amber-100 hover:bg-amber-200 text-amber-800 rounded transition-colors shrink-0"
                  >
                    Copy
                  </button>
                </div>
              </div>
            )}

            <div className="mb-2">
              <label className="block text-[11px] font-medium text-slate-500 mb-1">
                Device Preference
              </label>
              <UISelect
                value={devicePref}
                onChange={handleDeviceChange}
                options={[
                  {
                    value: "auto",
                    label: deviceInfo?.gpu_available
                      ? `Auto (GPU: ${deviceInfo.gpu_name})`
                      : deviceInfo?.cpu_count && deviceInfo.cpu_count > 1
                      ? `Auto (Multi-Threaded CPU - ${deviceInfo.cpu_count} Threads)`
                      : "Auto (CPU)",
                  },
                  {
                    value: "gpu",
                    label: `GPU Acceleration ${deviceInfo?.gpu_available ? `(${deviceInfo.gpu_name})` : "[Unavailable]"}`,
                  },
                  {
                    value: "multithread_cpu",
                    label: `Multi-Threaded CPU Engine (${deviceInfo?.cpu_count || 4} Threads)`,
                  },
                  {
                    value: "cpu",
                    label: "Single-Threaded CPU Engine (1 Thread)",
                  },
                ]}
                placeholder="Select device"
              />
            </div>

            <div className="mb-2">
              <label className="block text-[11px] font-medium text-slate-500 mb-1">
                Batch Concurrency
              </label>
              <UISelect
                value={String(concurrency)}
                onChange={(val) => handleConcurrencyChange(Number(val))}
                options={[
                  {
                    value: "1",
                    label: "1 Worker (Sequential - Safe)",
                  },
                  {
                    value: "2",
                    label: "2 Workers (Dual Parallel)",
                  },
                  {
                    value: "3",
                    label: "3 Workers (Tri Parallel)",
                  },
                  {
                    value: "4",
                    label: "4 Workers (Quad Parallel - Fast)",
                  },
                ]}
                placeholder="Select concurrency"
              />
            </div>

            <div className="text-[10px] text-slate-400 leading-tight pt-1 border-t border-slate-100">
              {deviceInfo?.gpu_available
                ? "Tensor operations accelerated natively via local GPU execution provider."
                : "Parallel tensor operations accelerated across all CPU cores via ONNX Runtime."}
            </div>
          </div>
        </div>

        {/* Right Column: Download or Upload/Samples */}
        <div className="portal-content-col">
          {loadError && (
            <div className="locked-notice shrink-0" style={{ background: "#fef2f2", borderColor: "#fecaca", color: "#dc2626", marginBottom: "0.75rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <AlertCircle size={16} />
                <span>{loadError}</span>
              </div>
            </div>
          )}

          {!allDownloaded && isBackendConnected ? (
            <div className="flex flex-col h-full overflow-hidden">
              <div className="shrink-0">
                <h3 style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--color-navy)", marginBottom: "0.5rem" }}>
                  Foundation Models Required
                </h3>
                <div className="locked-notice">
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <AlertCircle size={16} />
                    <span>Action Required: Please download foundation models to unlock imagery upload & analysis.</span>
                  </div>
                </div>
              </div>

              <div className="portal-scroll-area flex-1 min-h-0">
                <div className="models-required-list">
                  {MODEL_GROUPS.map((group) => {
                    const ready = isGroupReady(group.id);
                    return (
                      <div key={group.id} className="model-item">
                        <div className="model-item-info">
                          <div className="model-name">{group.name}</div>
                          <div className="model-meta">
                            <span className="model-task">{group.task}</span>
                            <span className="model-size-badge">{group.size}</span>
                          </div>
                        </div>
                        <div className="model-item-actions">
                          <span
                            className={ready ? "model-badge-cached" : "model-badge-needed"}
                          >
                            {ready ? (
                              <>
                                <Check size={11} className="badge-icon" />
                                <span>Downloaded</span>
                              </>
                            ) : (
                              <>
                                <DownloadCloud size={11} className="badge-icon" />
                                <span>Download Required</span>
                              </>
                            )}
                          </span>
                          {ready ? (
                            <button
                              type="button"
                              className="btn-model-action danger"
                              onClick={() => handleDeleteModel(group.id)}
                              disabled={downloadProgress.is_downloading || isDeletingModel === group.id}
                              title={`Delete ${group.name}`}
                            >
                              <Trash2 size={12} className={isDeletingModel === group.id ? "spin" : ""} />
                            </button>
                          ) : (
                            <button
                              type="button"
                              className="btn-model-action primary"
                              onClick={() => handleStartDownload(group.id)}
                              disabled={downloadProgress.is_downloading || !!isDeletingModel}
                              title={`Download ${group.name}`}
                            >
                              <DownloadCloud
                                size={12}
                                className={
                                  downloadProgress.is_downloading &&
                                  downloadProgress.current_file.includes(group.name)
                                    ? "spin"
                                    : ""
                                }
                              />
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="shrink-0 pt-3 mt-auto border-t border-slate-100 bg-white">
                <button
                  className="btn btn-primary btn-block"
                  style={{ padding: "12px" }}
                  onClick={() => handleStartDownload()}
                  disabled={downloadProgress.is_downloading || !!isDeletingModel}
                >
                  <DownloadCloud size={18} />
                  {downloadProgress.is_downloading
                    ? "Downloading Weights..."
                    : "Download Foundation Models Now"}
                </button>
              </div>

              {downloadProgress.is_downloading && (
                <div className="shrink-0 mt-2">
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: 600, marginBottom: 4 }}>
                    <span>{downloadProgress.current_file}</span>
                    <span>{downloadProgress.overall_pct}%</span>
                  </div>
                  <Progress
                    value={downloadProgress.overall_pct}
                    className="h-2 bg-slate-100 my-1"
                    indicatorColor="bg-[#0d7c85]"
                  />
                  <div style={{ fontSize: "0.76rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                    {downloadProgress.detail}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col h-full overflow-hidden">
              {/* Modern Segmented Tab Bar */}
              <div className="shrink-0 mb-2">
                <div className="inline-flex p-1 rounded-xl bg-slate-100/90 border border-slate-200/80 w-fit">
                  <button
                    type="button"
                    className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeTab === "upload"
                        ? "bg-white text-slate-900 shadow-xs"
                        : "text-slate-500 hover:text-slate-800"
                    }`}
                    onClick={() => setActiveTab("upload")}
                  >
                    <UploadCloud size={14} className={activeTab === "upload" ? "text-teal-600" : ""} />
                    <span>Upload Images</span>
                    {stagedImages.length > 0 && (
                      <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[10px] bg-teal-100 text-teal-700 font-mono">
                        {stagedImages.length}
                      </span>
                    )}
                  </button>
                  <button
                    type="button"
                    className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeTab === "samples"
                        ? "bg-white text-slate-900 shadow-xs"
                        : "text-slate-500 hover:text-slate-800"
                    }`}
                    onClick={() => setActiveTab("samples")}
                  >
                    <FolderHeart size={14} className={activeTab === "samples" ? "text-teal-600" : ""} />
                    <span>Sample Library</span>
                    <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[10px] bg-slate-200/70 text-slate-600 font-mono">
                      {availableSamples.length || 24}
                    </span>
                  </button>
                  <button
                    type="button"
                    className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeTab === "video"
                        ? "bg-white text-slate-900 shadow-xs"
                        : "text-slate-500 hover:text-slate-800"
                    }`}
                    onClick={() => setActiveTab("video")}
                  >
                    <Film size={14} className={activeTab === "video" ? "text-cyan-600" : ""} />
                    <span>Video Transect</span>
                    {videoJobStatus && (videoJobStatus.state === "processing" || videoJobStatus.state === "paused") ? (
                      <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[9px] bg-cyan-500 text-white font-bold animate-pulse">
                        {videoJobStatus.progress_pct}%
                      </span>
                    ) : (
                      <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[9px] bg-cyan-100 text-cyan-800 font-bold uppercase tracking-wider">
                        Live
                      </span>
                    )}
                  </button>
                </div>
              </div>

              {/* Ongoing Video Transect Alert Banner (when viewing other tabs) */}
              {videoJobStatus && (videoJobStatus.state === "processing" || videoJobStatus.state === "paused") && activeTab !== "video" && (
                <div className="shrink-0 mb-2 px-3.5 py-2 rounded-xl bg-cyan-50/90 border border-cyan-200 text-cyan-900 text-xs flex items-center justify-between shadow-xs">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="relative flex h-2 w-2 shrink-0">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-600" />
                    </span>
                    <span className="font-bold truncate">Video Transect Ongoing:</span>
                    <span className="text-slate-600 truncate max-w-[200px]">{videoJobStatus.filename}</span>
                    <span className="font-mono text-[11px] text-cyan-800 bg-white border border-cyan-200 px-1.5 py-0.2 rounded shrink-0">
                      Frame {videoJobStatus.current_frame} / {videoJobStatus.total_frames} ({videoJobStatus.progress_pct}%)
                    </span>
                    {videoJobStatus.current_model && (
                      <span className="text-[10px] bg-cyan-100/90 text-cyan-800 font-semibold px-1.5 py-0.2 rounded shrink-0">
                        {videoJobStatus.current_model}
                      </span>
                    )}
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setActiveTab("video");
                    }}
                    className="h-6 px-2.5 text-[10px] font-bold border-cyan-300 text-cyan-800 bg-white hover:bg-cyan-100 gap-1 shrink-0"
                  >
                    <span>View Live Transect</span>
                    <ArrowRight size={10} />
                  </Button>
                </div>
              )}

              {/* Tab 1: Upload */}
              {activeTab === "upload" && (
                <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
                  <div className="shrink-0 mb-2">
                    <h4 className="text-sm font-bold text-[#0f1e4a] tracking-tight">
                      Upload Coral Imagery
                    </h4>
                    <p className="text-[11px] text-slate-500">
                      Import underwater survey photos or a .zip archive for batch instance segmentation
                    </p>
                  </div>

                  <div className="portal-scroll-area flex-1 min-h-0">
                    <div
                      className={`dropzone ${dragActive ? "drag-active" : ""}`}
                      onDragOver={onDragOver}
                      onDragLeave={onDragLeave}
                      onDrop={onDrop}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <UploadCloud size={30} className="dropzone-icon text-teal-600 mb-1.5" />
                      <div className="dropzone-text text-xs font-semibold text-slate-700">
                        Select or drag & drop coral images or a .zip archive
                      </div>
                      <div className="dropzone-hint text-[11px] text-slate-400 mt-0.5">
                        Supported formats: .jpg, .jpeg, .png, .webp, .zip
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept=".jpg,.jpeg,.png,.webp,.zip"
                        style={{ display: "none" }}
                        onChange={(e) => {
                          if (e.target.files && e.target.files.length > 0) {
                            handleFiles(e.target.files);
                          }
                        }}
                      />
                    </div>

                    {stagedImages.length > 0 && (
                      <div className="mt-2.5">
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="text-xs font-semibold text-emerald-700 flex items-center gap-1.5">
                            <CheckCircle2 size={13} />
                            <span>{stagedImages.length} image(s) ready for analysis</span>
                          </div>
                          <button
                            type="button"
                            className="text-[11px] font-medium text-slate-400 hover:text-red-600 transition-colors"
                            onClick={() => setStagedImages([])}
                          >
                            Clear all
                          </button>
                        </div>

                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 my-1">
                          {stagedImages.map((img, idx) => (
                            <div
                              key={idx}
                              className="group relative rounded-xl border border-slate-200/90 bg-white overflow-hidden shadow-xs hover:shadow-md hover:border-teal-500/60 transition-all duration-200 flex flex-col"
                            >
                              <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-100">
                                <img
                                  src={img.dataUrl}
                                  alt={img.name}
                                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 ease-out"
                                />
                                <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded-md bg-slate-900/70 backdrop-blur-md text-white font-mono text-[10px] font-semibold tracking-wider shadow-xs">
                                  #{String(idx + 1).padStart(2, "0")}
                                </div>
                                <button
                                  type="button"
                                  className="absolute top-2 right-2 h-5 w-5 rounded-full bg-slate-900/70 hover:bg-red-600 text-white flex items-center justify-center transition-colors shadow-xs"
                                  onClick={() => setStagedImages((prev) => prev.filter((_, i) => i !== idx))}
                                  title="Remove image"
                                >
                                  <X size={11} strokeWidth={2.5} />
                                </button>
                              </div>
                              <div className="px-2 py-1.5 bg-white flex items-center justify-between gap-1 border-t border-slate-100">
                                <span
                                  className="text-[11px] font-medium text-slate-700 truncate"
                                  title={img.name}
                                >
                                  {img.name}
                                </span>
                                <span className="text-[9px] font-mono uppercase text-slate-400 bg-slate-50 border border-slate-200/60 px-1 py-0.2 rounded shrink-0">
                                  {img.name.split(".").pop() || "IMG"}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {stagedImages.length > 0 &&
                    renderBatchControlPanel(stagedImages.length, handleStartBatchUpload, isStartingBatch)
                  }
                </div>
              )}

              {/* Tab 2: Samples */}
              {activeTab === "samples" && (
                <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
                  {/* Single-line Toolbar Header */}
                  <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 mb-2 pb-1.5 border-b border-slate-100">
                    <div>
                      <h4 className="text-sm font-bold text-[#0f1e4a] tracking-tight">
                        Select from Sample Library
                      </h4>
                      <p className="text-[11px] text-slate-500">
                        {sampleSubset.length} sample image{sampleSubset.length === 1 ? "" : "s"} selected for instance segmentation
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <label className="text-xs font-semibold text-slate-600">Batch:</label>
                      <div className="w-48">
                        <UISelect
                          value={selectedPreset}
                          onChange={setSelectedPreset}
                          options={[
                            { value: "First 6 Samples", label: "First 6 Samples" },
                            { value: "First 12 Samples", label: "First 12 Samples" },
                            { value: "First 24 Samples", label: "First 24 Samples" },
                            { value: "All Available Samples", label: "All Available Samples" },
                            { value: "Select Specific Files", label: "Select Specific Files" },
                          ]}
                          placeholder="Select batch preset"
                        />
                      </div>
                    </div>
                  </div>

                  {selectedPreset === "Select Specific Files" && (
                    <div className="shrink-0 form-group mb-2">
                      <label className="form-label text-xs font-semibold text-slate-600">Choose Files</label>
                      <div className="max-h-28 overflow-y-auto border border-slate-200 rounded-lg p-2 bg-slate-50/50 space-y-1">
                        {availableSamples.map((s) => (
                          <label key={s.filename} className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer hover:bg-white p-1 rounded">
                            <input
                              type="checkbox"
                              className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                              checked={selectedSpecific.includes(s.filename)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedSpecific((prev) => [...prev, s.filename]);
                                } else {
                                  setSelectedSpecific((prev) => prev.filter((fn) => fn !== s.filename));
                                }
                              }}
                            />
                            <span className="font-mono text-[11px]">{s.filename}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Scrollable Imagery Container - Image Cards Grid */}
                  <div className="portal-scroll-area flex-1 min-h-0">
                    {!isBackendConnected ? (
                      <div className="p-8 text-center bg-slate-50 rounded-xl border border-dashed border-slate-200 text-slate-500 text-xs my-4 flex flex-col items-center gap-2.5">
                        <div className="flex items-center gap-2 font-semibold text-slate-700 text-sm">
                          <Loader2 size={16} className="animate-spin text-teal-600" />
                          <span>Connecting to Python Vision Engine...</span>
                        </div>
                        <p className="text-slate-500 max-w-md text-xs">
                          Please ensure the backend is started locally (<code>./start_backend.sh</code>) or connect to your Google Colab / remote tunnel instance.
                        </p>
                        <div className="flex items-center gap-2 mt-1">
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="h-7 text-xs border-slate-300"
                            onClick={handleManualRetry}
                          >
                            Retry Connection
                          </Button>
                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            className="h-7 text-xs bg-teal-50 text-teal-700 hover:bg-teal-100 border border-teal-200"
                            onClick={() => setShowRemoteModal(true)}
                          >
                            Set Google Colab / Remote URL
                          </Button>
                        </div>
                      </div>
                    ) : sampleSubset.length === 0 ? (
                      <div className="p-8 text-center bg-slate-50 rounded-xl border border-dashed border-slate-200 text-slate-500 text-xs my-4">
                        No sample images found. Please verify <code>demo_images/</code> folder.
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 my-1">
                        {sampleSubset.map((s, idx) => (
                          <div
                            key={idx}
                            className="group relative rounded-xl border border-slate-200/90 bg-white overflow-hidden shadow-xs hover:shadow-md hover:border-teal-500/60 transition-all duration-200 flex flex-col"
                          >
                            <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-100">
                              <img
                                src={getSampleImageUrl(s.filename)}
                                alt={s.filename}
                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 ease-out"
                                onError={(e) => {
                                  (e.target as HTMLImageElement).src =
                                    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'><rect width='100' height='100' fill='%23eee'/><text x='50%' y='50%' text-anchor='middle' fill='%23aaa' dy='.3em'>Image</text></svg>";
                                }}
                              />
                              {/* Floating index badge top-left */}
                              <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded-md bg-slate-900/70 backdrop-blur-md text-white font-mono text-[10px] font-semibold tracking-wider shadow-xs">
                                #{String(idx + 1).padStart(2, "0")}
                              </div>
                              {/* Selected indicator top-right */}
                              <div className="absolute top-2 right-2 h-5 w-5 rounded-full bg-emerald-500/90 text-white flex items-center justify-center shadow-xs">
                                <Check size={11} strokeWidth={3} />
                              </div>
                            </div>

                            {/* Clean metadata caption bar */}
                            <div className="px-2 py-1.5 bg-white flex items-center justify-between gap-1 border-t border-slate-100">
                              <span
                                className="text-[11px] font-medium text-slate-700 truncate"
                                title={s.filename}
                              >
                                {s.filename}
                              </span>
                              <span className="text-[9px] font-mono uppercase text-slate-400 bg-slate-50 border border-slate-200/60 px-1 py-0.2 rounded shrink-0">
                                {s.filename.split(".").pop() || "PNG"}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Fixed Bottom Launch Bar - Go Button ALWAYS visible */}
                  {renderBatchControlPanel(sampleSubset.length, handleStartBatchSamples, isStartingBatch)}
                </div>
              )}

              {/* Tab 3: Video Transect Studio */}
              {activeTab === "video" && (
                <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
                  <div className="shrink-0 mb-2 flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-bold text-[#0f1e4a] tracking-tight">
                        Video Transect Studio
                      </h4>
                      <p className="text-[11px] text-slate-500">
                        Autonomous benthic colony tracking across underwater ROV, AUV, and diver videos
                      </p>
                    </div>

                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={handleLoadSampleVideo}
                      disabled={isUploadingVideo || isStartingVideoJob}
                      className="h-7 px-2.5 text-xs font-semibold gap-1.5 border-cyan-300 text-cyan-800 bg-cyan-50/60 hover:bg-cyan-100"
                    >
                      <Sparkles size={12} className="text-cyan-600" />
                      <span>Load Sample Video</span>
                    </Button>
                  </div>

                  {videoError && (
                    <div className="mb-2 p-2 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
                      <AlertCircle size={14} className="shrink-0" />
                      <span>{videoError}</span>
                    </div>
                  )}

                  <div className="portal-scroll-area flex-1 min-h-0 space-y-3">
                    {/* Hidden video input */}
                    <input
                      ref={videoInputRef}
                      type="file"
                      accept=".mp4,.mov,.avi,.mkv,.webm"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleVideoFileSelect(file);
                      }}
                    />

                    {/* Drag and Drop Box */}
                    <div
                      className={`dropzone cursor-pointer ${videoDragActive ? "drag-active" : ""}`}
                      onDragOver={(e) => {
                        e.preventDefault();
                        setVideoDragActive(true);
                      }}
                      onDragLeave={() => setVideoDragActive(false)}
                      onDrop={(e) => {
                        e.preventDefault();
                        setVideoDragActive(false);
                        const file = e.dataTransfer.files?.[0];
                        if (file) handleVideoFileSelect(file);
                      }}
                      onClick={() => videoInputRef.current?.click()}
                    >
                      <Film size={32} className="dropzone-icon text-cyan-600 mb-1.5" />
                      <div className="dropzone-text text-xs font-semibold text-slate-700">
                        Select or drag & drop underwater video (.mp4, .mov, .avi)
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">
                        High-FPS tracking with ego-motion compensation & persistent colony IDs
                      </div>
                    </div>

                    {/* Selected Video Card Preview */}
                    {uploadedVideo && (
                      <div className="rounded-xl border border-cyan-200 bg-cyan-50/30 p-3.5 space-y-3 shadow-xs">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2.5">
                            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-700">
                              <Video size={18} />
                            </div>
                            <div>
                              <div className="text-xs font-bold text-slate-800 truncate max-w-[280px]">
                                {uploadedVideo.video_name}
                              </div>
                              <div className="text-[11px] text-slate-500 font-mono">
                                {uploadedVideo.metadata.width}x{uploadedVideo.metadata.height} •{" "}
                                {uploadedVideo.metadata.fps} FPS •{" "}
                                {uploadedVideo.metadata.duration_sec}s ({uploadedVideo.metadata.total_frames} frames)
                              </div>
                            </div>
                          </div>

                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setUploadedVideo(null)}
                            className="h-7 w-7 p-0 text-slate-400 hover:text-red-600"
                          >
                            <X size={14} />
                          </Button>
                        </div>

                        {/* Survey Cadence Status */}
                        <div className="space-y-1.5 pt-1 border-t border-cyan-100">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] font-semibold text-slate-700">
                              Survey Cadence Preset
                            </span>
                            <span className="text-[10px] text-cyan-700 font-medium bg-cyan-100/70 px-1.5 py-0.5 rounded">
                              Max Accuracy
                            </span>
                          </div>
                          <div className="p-2 rounded-lg border border-cyan-200 bg-cyan-50/50 text-left">
                            <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                              <span className="w-1.5 h-1.5 rounded-full bg-cyan-600" />
                              High Accuracy Cadence
                            </div>
                            <div className="text-[10px] text-slate-500 mt-0.5">
                              Dense 10-25 frame keyframe census with 70% benthic overlap gate
                            </div>
                          </div>
                        </div>

                        {/* Tracker Engine Selector */}
                        <div className="space-y-1.5 pt-1 border-t border-cyan-100">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] font-semibold text-slate-700">
                              Tracking Engine
                            </span>
                            <span className="text-[10px] text-cyan-700 font-mono">
                              {videoTrackerMode === "sam2" ? "Neural Spatio-Temporal Masks" : "Colony-Level Flow + Re-ID"}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <button
                              type="button"
                              onClick={() => setVideoTrackerMode("sam2")}
                              className={`p-2 rounded-lg border text-left transition-all ${
                                videoTrackerMode === "sam2"
                                  ? "bg-cyan-50/80 border-cyan-500 ring-1 ring-cyan-500/30"
                                  : "bg-white/60 border-slate-200 hover:border-slate-300"
                              }`}
                            >
                              <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                                SAM 2 Neural Tracker
                              </div>
                              <div className="text-[10px] text-slate-500 mt-0.5">
                                Deformable masks with memory attention (Max Accuracy)
                              </div>
                            </button>

                            <button
                              type="button"
                              onClick={() => setVideoTrackerMode("optical_flow")}
                              className={`p-2 rounded-lg border text-left transition-all ${
                                videoTrackerMode === "optical_flow"
                                  ? "bg-cyan-50/80 border-cyan-500 ring-1 ring-cyan-500/30"
                                  : "bg-white/60 border-slate-200 hover:border-slate-300"
                              }`}
                            >
                              <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                Colony Flow + Re-ID
                              </div>
                              <div className="text-[10px] text-slate-500 mt-0.5">
                                Per-colony optical flow with depth compensation (High Speed)
                              </div>
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Fixed Bottom Launch & Transect Control Bar */}
                  {renderVideoControlPanel()}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Model Management Modal */}
      <Dialog open={showDeleteModal} onOpenChange={setShowDeleteModal}>
        <DialogContent className="max-w-md p-6">
          <DialogHeader>
            <div className="flex items-center gap-2">
              <Trash2 className="h-5 w-5 text-red-500" />
              <DialogTitle className="text-base font-bold text-[#0f1e4a]">
                Manage Foundation Models
              </DialogTitle>
            </div>
            <DialogDescription className="text-xs text-slate-500 pt-1">
              Delete individual models to free disk space. You can re-download any deleted model at any time.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-2 my-2 max-h-60 overflow-y-auto">
            {MODEL_GROUPS.map((group) => {
              const ready = isGroupReady(group.id);
              return (
                <div
                  key={group.id}
                  className="flex items-center justify-between p-2.5 rounded-lg border border-slate-100 bg-slate-50/60"
                >
                  <div>
                    <div className="text-xs font-bold text-slate-800">{group.name}</div>
                    <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
                      <span>{group.task}</span>
                      <span>•</span>
                      <span>{group.size}</span>
                    </div>
                  </div>
                  <div>
                    {ready ? (
                      <Button
                        type="button"
                        variant="destructive"
                        size="sm"
                        className="h-7 px-2.5 text-xs gap-1.5 rounded-md"
                        onClick={() => handleDeleteModel(group.id)}
                        disabled={!!isDeletingModel}
                        title={`Delete ${group.name}`}
                      >
                        <Trash2 className={`h-3.5 w-3.5 ${isDeletingModel === group.id ? "animate-spin" : ""}`} />
                        <span>{isDeletingModel === group.id ? "Deleting..." : "Delete"}</span>
                      </Button>
                    ) : (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 px-2.5 text-xs gap-1.5 rounded-md border-teal-200 text-teal-700 hover:bg-teal-50"
                        onClick={() => {
                          handleStartDownload(group.id);
                          setShowDeleteModal(false);
                        }}
                        disabled={downloadProgress.is_downloading || !!isDeletingModel}
                        title={`Download ${group.name}`}
                      >
                        <DownloadCloud className="h-3.5 w-3.5" />
                        <span>Download</span>
                      </Button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <DialogFooter className="mt-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowDeleteModal(false)}
              disabled={!!isDeletingModel}
            >
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Remote / Google Colab Backend URL Modal */}
      <Dialog open={showRemoteModal} onOpenChange={setShowRemoteModal}>
        <DialogContent className="max-w-md p-6">
          <DialogHeader>
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-teal-600" />
              <DialogTitle className="text-base font-bold text-[#0f1e4a]">
                Connect to Remote / Colab Backend
              </DialogTitle>
            </div>
            <DialogDescription className="text-xs text-slate-500 pt-1">
              Connect this desktop app to a Google Colab GPU instance or remote server exposed via Cloudflare Tunnel or ngrok.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3 my-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Backend Server URL
              </label>
              <input
                type="text"
                value={remoteUrlInput}
                onChange={(e) => setRemoteUrlInput(e.target.value)}
                placeholder="https://your-tunnel-name.trycloudflare.com"
                className="w-full text-xs px-3 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500/20 font-mono"
              />
              <p className="text-[10px] text-slate-500 mt-1">
                Leave empty or click Reset to use the local machine backend (<code>http://127.0.0.1:8000</code>).
              </p>
            </div>
          </div>

          <DialogFooter className="flex items-center justify-between sm:justify-between mt-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleResetToLocal}
              className="text-xs text-slate-600 hover:bg-slate-100"
            >
              Reset to Local
            </Button>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setShowRemoteModal(false)}
                className="text-xs"
              >
                Cancel
              </Button>
              <Button
                type="button"
                size="sm"
                onClick={handleSaveRemoteUrl}
                className="text-xs bg-teal-600 hover:bg-teal-700 text-white font-medium"
              >
                Save & Connect
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ExportDatasetModal
        open={isExportDatasetModalOpen}
        onOpenChange={setIsExportDatasetModalOpen}
      />
    </div>
  );
};
