import React, { useState, useEffect } from "react";
import {
  AlertCircle,
  Play,
  Pause,
  StopCircle,
  FolderArchive,
  Loader2,
  CheckCircle2,
  Cpu,
} from "lucide-react";
import { Sidebar } from "../components/Sidebar";
import { TopBar } from "../components/TopBar";
import { PaginationBar } from "../components/PaginationBar";
import { ImageViewer } from "../components/ImageViewer";
import { MetricCards } from "../components/MetricCards";
import { SegmentsTable } from "../components/SegmentsTable";
import { ExportPanel } from "../components/ExportPanel";
import { ExportDatasetModal } from "../components/ExportDatasetModal";
import { SegmentationEditorModal } from "../components/SegmentationEditorModal";
import { Button } from "@/components/ui/button";
import {
  LoadedImage,
  DeviceInfo,
  CoralSegment,
  SummaryStats,
  HealthSummary,
  AnalysisStage,
  BatchStatusResponse,
} from "../types";
import {
  runSegmentation,
  runEnrichment,
  renderOverlay,
  getDevice,
  setDevicePreference,
  getBatchStatus,
  pauseBatch,
  resumeBatch,
  cancelBatch,
  prioritizeBatchImage,
  getPrecomputedResult,
} from "../services/api";

interface AnalysisViewProps {
  images: LoadedImage[];
  onBackToUpload: () => void;
}

export const AnalysisView: React.FC<AnalysisViewProps> = ({
  images,
  onBackToUpload,
}) => {
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);

  // Inference Parameters
  const [pointsPerSide, setPointsPerSide] = useState<number>(16);
  const [iouThresh, setIouThresh] = useState<number>(0.50);
  const [stabilityThresh, setStabilityThresh] = useState<number>(0.50);
  const [minAreaPx, setMinAreaPx] = useState<number>(100);

  // Display Controls
  const [layoutMode, setLayoutMode] = useState<string>("Side-by-Side");
  const [colorMode, setColorMode] = useState<string>("instance");
  const [alpha, setAlpha] = useState<number>(0.45);
  const [drawContours, setDrawContours] = useState<boolean>(true);
  const [drawLabels, setDrawLabels] = useState<boolean>(true);
  const [drawBoxes, setDrawBoxes] = useState<boolean>(false);
  const [selectedMaskId, setSelectedMaskId] = useState<number | null>(null);

  // Device Info
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [devicePref, setDevicePref] = useState<string>("auto");

  // Analysis State
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysisStage, setAnalysisStage] = useState<AnalysisStage>("idle");
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [overlaySrc, setOverlaySrc] = useState<string | null>(null);
  const [segments, setSegments] = useState<CoralSegment[]>([]);
  const [stats, setStats] = useState<SummaryStats>({
    total_corals_detected: 0,
    coral_coverage_pct: 0,
    coral_covered_pixels: 0,
    total_image_pixels: 0,
    image_resolution: "0x0",
    mean_iou_confidence: 0,
    mean_stability_score: 0,
  });
  const [healthSummary, setHealthSummary] = useState<HealthSummary>({
    healthy_count: 0,
    bleached_count: 0,
    bleaching_prevalence_pct: 0,
  });
  const [sceneEval, setSceneEval] = useState<Record<string, any>>({});

  // Background Batch Tracking & Dataset Export
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null);
  const [isExportDatasetModalOpen, setIsExportDatasetModalOpen] = useState<boolean>(false);
  const [isCancellingBatch, setIsCancellingBatch] = useState<boolean>(false);
  const [batchFeedback, setBatchFeedback] = useState<string | null>(null);

  // Manual Segmentation & Annotation Studio Modal State
  const [isEditorOpen, setIsEditorOpen] = useState<boolean>(false);

  const handleEditorSaveSuccess = (data: {
    summary: SummaryStats;
    segments: CoralSegment[];
    health_summary: HealthSummary;
    overlay_base64: string;
  }) => {
    setStats(data.summary);
    setSegments(data.segments);
    setHealthSummary(data.health_summary);
    setOverlaySrc(data.overlay_base64);
  };

  const currentImage = images[currentIndex] || images[0];

  // Poll batch status every 1000ms
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let isSubscribed = true;

    const poll = async () => {
      try {
        const res = await getBatchStatus();
        if (isSubscribed) {
          setBatchStatus(res);
        }
      } catch {
        // silent
      }
      if (isSubscribed) {
        timer = setTimeout(poll, 1000);
      }
    };

    poll();
    return () => {
      isSubscribed = false;
      clearTimeout(timer);
    };
  }, []);

  // Fetch initial device info
  useEffect(() => {
    getDevice()
      .then(setDeviceInfo)
      .catch((e: unknown) => console.error("Device fetch failed", e));
  }, []);

  const handleDeviceChange = async (val: string) => {
    setDevicePref(val);
    try {
      const updated = await setDevicePreference(val);
      setDeviceInfo(updated);
    } catch (e) {
      console.error(e);
    }
  };

  // Reset overlay & telemetry when switching images
  useEffect(() => {
    setOverlaySrc(null);
    setSelectedMaskId(null);
    setSegments([]);
    setStats({
      total_corals_detected: 0,
      coral_coverage_pct: 0,
      coral_covered_pixels: 0,
      total_image_pixels: 0,
      image_resolution: "0x0",
      mean_iou_confidence: 0,
      mean_stability_score: 0,
    });
  }, [currentImage?.name]);

  // Check Precomputed Results or Run Analysis on Image Change
  useEffect(() => {
    if (!currentImage) return;
    let isCancelled = false;

    const executeAnalysisOrLoadCache = async () => {
      // 1. Check if background batch already computed this image with default settings
      const isDefaultParams =
        pointsPerSide === 16 &&
        iouThresh === 0.50 &&
        stabilityThresh === 0.50 &&
        minAreaPx === 100;

      if (isDefaultParams) {
        try {
          const cached = await getPrecomputedResult(currentImage.name);
          if (isCancelled) return;

          if (cached && cached.status === "completed" && cached.segments) {
            setSegments(cached.segments);
            if (cached.summary) setStats(cached.summary);
            if (cached.health_summary) setHealthSummary(cached.health_summary);
            if (cached.scene_eval) setSceneEval(cached.scene_eval);
            if (cached.overlay_base64) setOverlaySrc(cached.overlay_base64);
            setIsLoading(false);
            setAnalysisStage("idle");
            return;
          } else if (cached && (cached.status === "processing" || cached.status === "pending")) {
            // Bump image to front of background processing queue
            prioritizeBatchImage(currentImage.name).catch(() => {});
            setIsLoading(true);
            setAnalysisStage((cached.stage as AnalysisStage) || "segmenting");
            return;
          }
        } catch {
          // fallback to manual run
        }
      }

      // 2. Direct run if not in batch or if user customized parameters
      setIsLoading(true);
      setAnalysisError(null);
      setAnalysisStage("segmenting");
      try {
        // Step 1: Run SAM ViT-B Segmentation
        await runSegmentation(
          currentImage.name,
          pointsPerSide,
          iouThresh,
          stabilityThresh,
          currentImage.dataUrl.startsWith("data:") ? currentImage.dataUrl : undefined
        );

        if (isCancelled) return;

        // Step 2: Enrich filtered masks with BioCLIP & YOLO11
        setAnalysisStage("classifying");
        const enrichResult = await runEnrichment(currentImage.name, minAreaPx);

        if (isCancelled) return;

        setSegments(enrichResult.segments);
        setStats(enrichResult.summary);
        setHealthSummary(enrichResult.health_summary);
        setSceneEval(enrichResult.scene_eval);

        // Step 3: Render Overlay
        setAnalysisStage("rendering");
        const overlayDataUrl = await renderOverlay({
          imageName: currentImage.name,
          minAreaPx,
          alpha,
          drawContours,
          drawLabels,
          drawBoxes,
          selectedMaskId,
          colorMode,
          layoutMode,
        });

        if (isCancelled) return;
        setOverlaySrc(overlayDataUrl);
      } catch (err: any) {
        console.error("Analysis execution error:", err);
        if (!isCancelled) {
          setAnalysisError(err?.message || "Analysis execution failed");
        }
      } finally {
        if (!isCancelled) {
          setIsLoading(false);
          setAnalysisStage("idle");
        }
      }
    };

    executeAnalysisOrLoadCache();

    return () => {
      isCancelled = true;
    };
  }, [
    currentImage?.name,
    pointsPerSide,
    iouThresh,
    stabilityThresh,
    minAreaPx,
  ]);

  // Auto-update active image as soon as background processing completes it
  useEffect(() => {
    if (!currentImage) return;
    const item = batchStatus?.items?.[currentImage.name];
    if (item && item.status === "completed" && (isLoading || segments.length === 0 || !overlaySrc)) {
      getPrecomputedResult(currentImage.name)
        .then((data) => {
          if (data && data.status === "completed" && data.segments) {
            setSegments(data.segments);
            if (data.summary) setStats(data.summary);
            if (data.health_summary) setHealthSummary(data.health_summary);
            if (data.scene_eval) setSceneEval(data.scene_eval);
            if (data.overlay_base64) setOverlaySrc(data.overlay_base64);
            setIsLoading(false);
            setAnalysisStage("idle");
          }
        })
        .catch(console.error);
    } else if (item && item.status === "processing" && !overlaySrc) {
      setIsLoading(true);
      setAnalysisStage((item.stage as AnalysisStage) || "segmenting");
    }
  }, [batchStatus, currentImage?.name, isLoading, segments.length, overlaySrc]);

  // Fast re-render overlay when display controls or selection change
  useEffect(() => {
    if (!currentImage || segments.length === 0) return;
    let isCancelled = false;

    const updateOverlay = async () => {
      try {
        const overlayDataUrl = await renderOverlay({
          imageName: currentImage.name,
          minAreaPx,
          alpha,
          drawContours,
          drawLabels,
          drawBoxes,
          selectedMaskId,
          colorMode,
          layoutMode,
        });
        if (!isCancelled) {
          setOverlaySrc(overlayDataUrl);
        }
      } catch (err) {
        console.error("Overlay update failed:", err);
      }
    };

    updateOverlay();

    return () => {
      isCancelled = true;
    };
  }, [
    alpha,
    drawContours,
    drawLabels,
    drawBoxes,
    selectedMaskId,
    colorMode,
    layoutMode,
  ]);

  // Global arrow navigation between images
  useEffect(() => {
    const handleKeyNav = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }
      if (e.key === "ArrowLeft") {
        setCurrentIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "ArrowRight") {
        setCurrentIndex((prev) => Math.min(prev + 1, images.length - 1));
      }
    };
    window.addEventListener("keydown", handleKeyNav);
    return () => window.removeEventListener("keydown", handleKeyNav);
  }, [images.length]);

  return (
    <div className="flex flex-col h-screen w-full overflow-hidden bg-background">
      {/* Top Bar: Back, Andromeida Branding, Hardware Device */}
      <TopBar onBack={onBackToUpload} deviceInfo={deviceInfo} />

      {/* Batch Processing Header Bar */}
      {batchStatus && (batchStatus.is_running || batchStatus.is_paused || batchStatus.completed > 0) && (
        <div className="flex flex-wrap items-center justify-between px-6 py-2 bg-slate-900 text-white border-b border-slate-800 text-xs gap-3 z-30 shadow-xs">
          <div className="flex items-center gap-2.5 min-w-0">
            {batchStatus.is_running && !batchStatus.is_paused && (
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-teal-400" />
              </span>
            )}
            {batchStatus.is_paused && <span className="h-2 w-2 rounded-full bg-amber-400" />}
            {!batchStatus.is_running && !batchStatus.is_paused && (
              <CheckCircle2 size={13} className="text-emerald-400" />
            )}

            <span className="font-semibold text-slate-100">
              {batchStatus.is_running
                ? (batchStatus.is_paused ? "Batch Paused" : "Autonomous Batch Processing")
                : "Batch Completed"}
            </span>

            <span className="font-mono text-[11px] text-teal-300 bg-teal-950/70 border border-teal-800/80 px-2 py-0.5 rounded-md">
              {batchStatus.completed} / {batchStatus.total} Processed ({batchStatus.percent}%)
            </span>

            {batchStatus.current_stage?.includes("Loading") && (
              <span className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-700 animate-pulse">
                <Cpu size={10} className="text-amber-400" />
                <span>Loading Model into RAM</span>
              </span>
            )}

            {batchStatus.is_running && !batchStatus.is_paused && (
              <span className="text-slate-400 text-[11px] truncate hidden md:inline">
                • {batchStatus.current_stage || (batchStatus.current_image ? `Analyzing ${batchStatus.current_image}` : "Processing")}
              </span>
            )}

            {(batchStatus.current_stage?.includes("Cancelled") || batchStatus.current_stage === "cancelled") && (
              <span className="text-red-400 text-[11px] truncate hidden md:inline">
                • Cancelled (Models unloaded from RAM)
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {(batchStatus.is_running || batchStatus.is_paused) && (
              <>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={isCancellingBatch}
                  onClick={async () => {
                    if (batchStatus.is_paused) {
                      const res = await resumeBatch();
                      setBatchStatus(res);
                    } else {
                      const res = await pauseBatch();
                      setBatchStatus(res);
                    }
                  }}
                  className="h-6 px-2 text-[10px] font-semibold gap-1 bg-slate-800 text-slate-200 border-slate-700 hover:bg-slate-700 hover:text-white"
                >
                  {batchStatus.is_paused ? <Play size={10} /> : <Pause size={10} />}
                  <span>{batchStatus.is_paused ? "Resume" : "Pause"}</span>
                </Button>

                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={isCancellingBatch}
                  onClick={async () => {
                    setIsCancellingBatch(true);
                    try {
                      const res = await cancelBatch();
                      setBatchStatus(res);
                      setBatchFeedback("Batch cancelled safely. Models unloaded from RAM.");
                      setTimeout(() => setBatchFeedback(null), 5000);
                    } catch (e: any) {
                      alert(`Error cancelling batch: ${e?.message || e}`);
                    } finally {
                      setIsCancellingBatch(false);
                    }
                  }}
                  className="h-6 px-2 text-[10px] font-semibold gap-1 bg-slate-800 text-red-400 border-red-900/60 hover:bg-red-950 hover:text-red-300"
                  title="Cancel processing and unload models from RAM"
                >
                  {isCancellingBatch ? (
                    <Loader2 size={10} className="animate-spin text-red-400" />
                  ) : (
                    <StopCircle size={10} />
                  )}
                  <span>{isCancellingBatch ? "Unloading..." : "Cancel"}</span>
                </Button>
              </>
            )}

            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setIsExportDatasetModalOpen(true)}
              className="h-6 px-2.5 text-[10px] font-bold gap-1 text-teal-300 bg-teal-950/80 border-teal-700/80 hover:bg-teal-900 hover:text-white rounded-md cursor-pointer"
              title="Export complete dataset across all processed images to custom folder or .zip archive"
            >
              <FolderArchive size={11} className="text-teal-300" />
              <span>Export Dataset</span>
            </Button>
          </div>
        </div>
      )}

      {batchFeedback && (
        <div className="px-6 py-1.5 bg-emerald-950 border-b border-emerald-800 text-emerald-300 text-xs font-semibold flex items-center justify-between">
          <span>{batchFeedback}</span>
          <button onClick={() => setBatchFeedback(null)} className="text-emerald-400 hover:text-white">✕</button>
        </div>
      )}

      {analysisError && (
        <div className="flex items-center justify-between px-6 py-2.5 bg-destructive/10 border-b border-destructive/20 text-destructive text-xs font-medium z-40">
          <div className="flex items-center gap-2">
            <AlertCircle className="size-4 shrink-0 text-destructive" />
            <span>{analysisError}</span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onBackToUpload}
            className="h-7 px-2.5 text-xs border-destructive/30 text-destructive hover:bg-destructive/15"
          >
            Return to Download Models
          </Button>
        </div>
      )}

      {/* Analysis Studio Body */}
      <div className="flex flex-1 overflow-hidden min-h-0 relative">
        {/* Collapsible Sidebar */}
        <Sidebar
          isCollapsed={isSidebarCollapsed}
          onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          onSetCollapsed={setIsSidebarCollapsed}
          pointsPerSide={pointsPerSide}
          onPointsPerSideChange={setPointsPerSide}
          iouThresh={iouThresh}
          onIouThreshChange={setIouThresh}
          stabilityThresh={stabilityThresh}
          onStabilityThreshChange={setStabilityThresh}
          minAreaPx={minAreaPx}
          onMinAreaPxChange={setMinAreaPx}
          layoutMode={layoutMode}
          onLayoutModeChange={setLayoutMode}
          colorMode={colorMode}
          onColorModeChange={setColorMode}
          alpha={alpha}
          onAlphaChange={setAlpha}
          drawContours={drawContours}
          onDrawContoursChange={setDrawContours}
          drawLabels={drawLabels}
          onDrawLabelsChange={setDrawLabels}
          drawBoxes={drawBoxes}
          onDrawBoxesChange={setDrawBoxes}
          segments={segments}
          selectedMaskId={selectedMaskId}
          onSelectMaskId={setSelectedMaskId}
          deviceInfo={deviceInfo}
          devicePreference={devicePref}
          onDevicePreferenceChange={handleDeviceChange}
        />

        {/* Main Studio Canvas & Results with Natural Rhythm */}
        <main className="flex-1 min-w-0 h-full overflow-y-auto min-h-0 bg-muted/20">
          <div className="max-w-7xl mx-auto w-full p-6 lg:p-8 flex flex-col gap-6">
            {/* Gallery Navigation Toolbar */}
            <PaginationBar
              currentIndex={currentIndex}
              totalImages={images.length}
              currentImageName={currentImage.name}
              imageNames={images.map((img) => img.name)}
              itemStatuses={
                batchStatus?.items
                  ? (Object.fromEntries(
                      Object.entries(batchStatus.items).map(([k, v]) => [k, v.status])
                    ) as any)
                  : undefined
              }
              onPrev={() => setCurrentIndex((prev) => Math.max(prev - 1, 0))}
              onNext={() => setCurrentIndex((prev) => Math.min(prev + 1, images.length - 1))}
              onSelectPage={(idx) => setCurrentIndex(idx)}
            />

            {/* Canvas Image Viewer with Hover Fullscreen & Coral Inspection */}
            <ImageViewer
              originalSrc={currentImage.dataUrl}
              overlaySrc={overlaySrc}
              layoutMode={layoutMode}
              colorMode={colorMode}
              isLoading={isLoading}
              analysisStage={analysisStage}
              segments={segments}
              imageResolution={stats.image_resolution}
              onOpenEditor={() => setIsEditorOpen(true)}
            />

            {/* 4 KPI Summary Cards with Consistent Height & Typography */}
            <MetricCards stats={stats} isLoading={isLoading} />

            {/* Detected Coral Segments Breakdown Table */}
            <SegmentsTable
              segments={segments}
              stats={stats}
              selectedSegmentId={selectedMaskId}
              onSelectSegment={(id) => setSelectedMaskId(id)}
              isLoading={isLoading}
              onOpenEditor={() => setIsEditorOpen(true)}
              onEditSegment={(id) => {
                setSelectedMaskId(id);
                setIsEditorOpen(true);
              }}
            />

            {/* Export & Data Inspector */}
            <ExportPanel
              imageName={currentImage.name}
              minAreaPx={minAreaPx}
              overlayDataUrl={overlaySrc}
              rawJsonData={{
                summary: stats,
                health_summary: healthSummary,
                scene_taxonomy: sceneEval?.taxonomy,
                scene_bleaching: sceneEval?.bleaching,
                segments: segments,
              }}
              hasSegments={segments.length > 0}
            />
          </div>
        </main>
      </div>

      <ExportDatasetModal
        open={isExportDatasetModalOpen}
        onOpenChange={setIsExportDatasetModalOpen}
        totalImagesCount={images.length}
      />

      <SegmentationEditorModal
        isOpen={isEditorOpen}
        onClose={() => setIsEditorOpen(false)}
        imageName={currentImage.name}
        imageSrc={currentImage.dataUrl}
        imageResolution={stats.image_resolution}
        initialSegments={segments}
        onSaveSuccess={handleEditorSaveSuccess}
        onNavigatePrev={() => setCurrentIndex((prev) => Math.max(prev - 1, 0))}
        onNavigateNext={() => setCurrentIndex((prev) => Math.min(prev + 1, images.length - 1))}
        hasPrev={currentIndex > 0}
        hasNext={currentIndex < images.length - 1}
      />
    </div>
  );
};
