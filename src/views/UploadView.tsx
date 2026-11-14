import React, { useState, useEffect, useMemo } from "react";
import {
  UploadCloud,
  FolderHeart,
  DownloadCloud,
  AlertCircle,
  Check,
  Trash2,
  ArrowRight,
  Film,
} from "lucide-react";
import JSZip from "jszip";
import { BrandLogo } from "../components/BrandLogo";
import { ExportDatasetModal } from "../components/ExportDatasetModal";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
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
import { MODEL_GROUPS, isGroupReady } from "../components/upload/modelGroups";
import { HardwareAccelerationCard } from "../components/upload/HardwareAccelerationCard";
import { SystemStatusSummary } from "../components/upload/SystemStatusSummary";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ImageUploadTab } from "../components/upload/ImageUploadTab";
import { SampleLibraryTab } from "../components/upload/SampleLibraryTab";
import { VideoTransectTab } from "../components/upload/VideoTransectTab";
import { ModelManagementModal } from "../components/upload/ModelManagementModal";
import { RemoteBackendModal } from "../components/upload/RemoteBackendModal";

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
  const [showSettings, setShowSettings] = useState<boolean>(false);

  // Video State
  const [uploadedVideo, setUploadedVideo] = useState<UploadedVideoItem | null>(activeVideo || null);
  const [videoJobStatus, setVideoJobStatus] = useState<VideoJobStatus | null>(null);
  const [isPausingResumingVideo, setIsPausingResumingVideo] = useState<boolean>(false);
  const [isUploadingVideo, setIsUploadingVideo] = useState<boolean>(false);
  const [isStartingVideoJob, setIsStartingVideoJob] = useState<boolean>(false);
  const [videoError, setVideoError] = useState<string | null>(null);

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

  // Trigger Download & Poll Progress
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
      setLoadError(
        "Process blocked: Foundation models are missing or deleted. You must download all models before proceeding."
      );
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
      setLoadError(
        "Process blocked: Foundation models are missing or deleted. You must download all models before proceeding."
      );
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

    try {
      const procRes = await processVideo({ video_name: uploadedVideo.video_name });

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

  return (
    <div className="portal-container">
      <div className="portal-card">
        {/* Left Column: brand + one-line system status (full configuration lives in Settings) */}
        <div className="portal-brand-col">
          <BrandLogo />
          <SystemStatusSummary
            isBackendConnected={isBackendConnected}
            isCheckingBackend={isCheckingBackend}
            allDownloaded={allDownloaded}
            deviceInfo={deviceInfo}
            onOpenSettings={() => setShowSettings(true)}
          />
        </div>

        {/* Right Column: Download or Upload/Samples */}
        <div className="portal-content-col">
          {loadError && (
            <div
              className="locked-notice shrink-0"
              style={{
                background: "#fef2f2",
                borderColor: "#fecaca",
                color: "#dc2626",
                marginBottom: "0.75rem",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <AlertCircle size={16} />
                <span>{loadError}</span>
              </div>
            </div>
          )}

          {!allDownloaded && isBackendConnected ? (
            <div className="flex flex-col h-full overflow-hidden">
              <div className="shrink-0">
                <h3
                  style={{
                    fontSize: "1.2rem",
                    fontWeight: 700,
                    color: "var(--color-navy)",
                    marginBottom: "0.5rem",
                  }}
                >
                  Foundation Models Required
                </h3>
                <div className="locked-notice">
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <AlertCircle size={16} />
                    <span>
                      Action Required: Please download foundation models to unlock imagery upload &
                      analysis.
                    </span>
                  </div>
                </div>
              </div>

              <div className="portal-scroll-area flex-1 min-h-0">
                <div className="models-required-list">
                  {MODEL_GROUPS.map((group) => {
                    const ready = isGroupReady(group.id, allDownloaded, modelsList);
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
                              disabled={
                                downloadProgress.is_downloading || isDeletingModel === group.id
                              }
                              title={`Delete ${group.name}`}
                            >
                              <Trash2
                                size={12}
                                className={isDeletingModel === group.id ? "spin" : ""}
                              />
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
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      marginBottom: 4,
                    }}
                  >
                    <span>{downloadProgress.current_file}</span>
                    <span>{downloadProgress.overall_pct}%</span>
                  </div>
                  <Progress
                    value={downloadProgress.overall_pct}
                    className="h-2 bg-slate-100 my-1"
                    indicatorColor="bg-[#0d7c85]"
                  />
                  <div
                    style={{
                      fontSize: "0.76rem",
                      color: "var(--color-text-muted)",
                      marginTop: 4,
                    }}
                  >
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
                    <UploadCloud
                      size={14}
                      className={activeTab === "upload" ? "text-teal-600" : ""}
                    />
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
                    <FolderHeart
                      size={14}
                      className={activeTab === "samples" ? "text-teal-600" : ""}
                    />
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
                    {videoJobStatus &&
                    (videoJobStatus.state === "processing" ||
                      videoJobStatus.state === "paused") ? (
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
              {videoJobStatus &&
                (videoJobStatus.state === "processing" || videoJobStatus.state === "paused") &&
                activeTab !== "video" && (
                  <div className="shrink-0 mb-2 px-3.5 py-2 rounded-xl bg-cyan-50/90 border border-cyan-200 text-cyan-900 text-xs flex items-center justify-between shadow-xs">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span className="relative flex h-2 w-2 shrink-0">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-600" />
                      </span>
                      <span className="font-bold truncate">Video Transect Ongoing:</span>
                      <span className="text-slate-600 truncate max-w-[200px]">
                        {videoJobStatus.filename}
                      </span>
                      <span className="font-mono text-[11px] text-cyan-800 bg-white border border-cyan-200 px-1.5 py-0.2 rounded shrink-0">
                        Frame {videoJobStatus.current_frame} / {videoJobStatus.total_frames} (
                        {videoJobStatus.progress_pct}%)
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
                <ImageUploadTab
                  stagedImages={stagedImages}
                  onFilesSelected={handleFiles}
                  onRemoveImage={(idx) =>
                    setStagedImages((prev) => prev.filter((_, i) => i !== idx))
                  }
                  onClearAll={() => setStagedImages([])}
                  onStartBatch={handleStartBatchUpload}
                  isStartingBatch={isStartingBatch}
                  batchStatus={batchStatus}
                  isBackendConnected={isBackendConnected}
                  exportFeedback={exportFeedback}
                  onClearExportFeedback={() => setExportFeedback(null)}
                  isCancellingBatch={isCancellingBatch}
                  onPauseResumeBatch={handlePauseResumeBatch}
                  onCancelBatch={handleCancelBatchRun}
                  onLaunchStudioNow={handleLaunchStudioNow}
                  onOpenExportModal={() => setIsExportDatasetModalOpen(true)}
                />
              )}

              {/* Tab 2: Samples */}
              {activeTab === "samples" && (
                <SampleLibraryTab
                  availableSamples={availableSamples}
                  sampleSubset={sampleSubset}
                  selectedPreset={selectedPreset}
                  onSelectPreset={setSelectedPreset}
                  selectedSpecific={selectedSpecific}
                  onToggleSpecific={(filename, checked) => {
                    if (checked) {
                      setSelectedSpecific((prev) => [...prev, filename]);
                    } else {
                      setSelectedSpecific((prev) => prev.filter((fn) => fn !== filename));
                    }
                  }}
                  isBackendConnected={isBackendConnected}
                  onManualRetry={handleManualRetry}
                  onOpenRemoteModal={() => setShowRemoteModal(true)}
                  onStartBatch={handleStartBatchSamples}
                  isStartingBatch={isStartingBatch}
                  batchStatus={batchStatus}
                  exportFeedback={exportFeedback}
                  onClearExportFeedback={() => setExportFeedback(null)}
                  isCancellingBatch={isCancellingBatch}
                  onPauseResumeBatch={handlePauseResumeBatch}
                  onCancelBatch={handleCancelBatchRun}
                  onLaunchStudioNow={handleLaunchStudioNow}
                  onOpenExportModal={() => setIsExportDatasetModalOpen(true)}
                />
              )}

              {/* Tab 3: Video Transect Studio */}
              {activeTab === "video" && (
                <VideoTransectTab
                  uploadedVideo={uploadedVideo}
                  onSelectVideoFile={handleVideoFileSelect}
                  onLoadSampleVideo={handleLoadSampleVideo}
                  onRemoveVideo={() => setUploadedVideo(null)}
                  videoError={videoError}
                  isUploadingVideo={isUploadingVideo}
                  isStartingVideoJob={isStartingVideoJob}
                  videoJobStatus={videoJobStatus}
                  isBackendConnected={isBackendConnected}
                  isPausingResumingVideo={isPausingResumingVideo}
                  onStartVideoAnalysis={handleStartVideoAnalysis}
                  onPauseResumeVideo={handlePauseResumeVideo}
                  onCancelVideo={handleCancelVideo}
                  onOpenVideoStudioFromPanel={handleOpenVideoStudioFromPanel}
                />
              )}
            </div>
          )}
        </div>
      </div>

      {/* Settings: compute device, concurrency and model management */}
      <Dialog open={showSettings} onOpenChange={setShowSettings}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Settings</DialogTitle>
          </DialogHeader>
          <HardwareAccelerationCard
            isBackendConnected={isBackendConnected}
            isCheckingBackend={isCheckingBackend}
            allDownloaded={allDownloaded}
            modelsList={modelsList}
            deviceInfo={deviceInfo}
            devicePref={devicePref}
            concurrency={concurrency}
            downloadProgress={downloadProgress}
            isDeletingModel={isDeletingModel}
            onManualRetry={handleManualRetry}
            onOpenRemoteModal={() => setShowRemoteModal(true)}
            onOpenDeleteModal={() => setShowDeleteModal(true)}
            onDeleteModel={handleDeleteModel}
            onDeviceChange={handleDeviceChange}
            onConcurrencyChange={handleConcurrencyChange}
          />
        </DialogContent>
      </Dialog>

      {/* Model Management Modal */}
      <ModelManagementModal
        open={showDeleteModal}
        onOpenChange={setShowDeleteModal}
        allDownloaded={allDownloaded}
        modelsList={modelsList}
        isDeletingModel={isDeletingModel}
        isDownloading={downloadProgress.is_downloading}
        onDeleteModel={handleDeleteModel}
        onStartDownload={handleStartDownload}
      />

      {/* Remote / Google Colab Backend URL Modal */}
      <RemoteBackendModal
        open={showRemoteModal}
        onOpenChange={setShowRemoteModal}
        remoteUrlInput={remoteUrlInput}
        onChangeRemoteUrl={setRemoteUrlInput}
        onSave={handleSaveRemoteUrl}
        onResetToLocal={handleResetToLocal}
      />

      <ExportDatasetModal
        open={isExportDatasetModalOpen}
        onOpenChange={setIsExportDatasetModalOpen}
      />
    </div>
  );
};
