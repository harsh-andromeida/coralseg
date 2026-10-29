import React, { useState, useEffect, useRef, useMemo } from "react";
import { TopBar } from "../components/TopBar";
import { Sidebar } from "../components/Sidebar";
import { VideoTimelineBar } from "../components/video/VideoTimelineBar";
import { VideoViewer } from "../components/video/VideoViewer";
import { MetricCards } from "../components/MetricCards";
import { SegmentsTable } from "../components/SegmentsTable";
import { VideoExportPanel } from "../components/video/VideoExportPanel";
import {
  UploadedVideoItem,
  VideoJobStatus,
  VideoTrackItem,
  DeviceInfo,
  CoralSegment,
  SummaryStats,
} from "../types";
import {
  getVideoStatus,
  getVideoFrameAnnotations,
  getDevice,
  setDevicePreference,
} from "../services/api";

interface VideoAnalysisViewProps {
  videoItem: UploadedVideoItem;
  initialJobId?: string;
  onBackToUpload: () => void;
}

export const VideoAnalysisView: React.FC<VideoAnalysisViewProps> = ({
  videoItem,
  initialJobId,
  onBackToUpload,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [jobId] = useState<string>(initialJobId || videoItem.job_id || "");
  const [jobStatus, setJobStatus] = useState<VideoJobStatus | null>(null);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [devicePref, setDevicePref] = useState<string>("auto");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);

  // Playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTimeSec, setCurrentTimeSec] = useState<number>(0);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);

  // Display Controls
  const [layoutMode, setLayoutMode] = useState<string>("Overlay Only");
  const [colorMode, setColorMode] = useState<string>("instance");
  const [alpha, setAlpha] = useState<number>(0.45);
  const [drawContours, setDrawContours] = useState<boolean>(true);
  const [drawLabels, setDrawLabels] = useState<boolean>(true);
  const [drawBoxes, setDrawBoxes] = useState<boolean>(false);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  // Threshold & Model Parameters (passed to Sidebar for complete consistency)
  const [pointsPerSide, setPointsPerSide] = useState<number>(12);
  const [iouThresh, setIouThresh] = useState<number>(0.45);
  const [stabilityThresh, setStabilityThresh] = useState<number>(0.45);
  const [minAreaPx, setMinAreaPx] = useState<number>(80);

  // Active frame tracks
  const [currentTracks, setCurrentTracks] = useState<VideoTrackItem[]>([]);

  const fps = videoItem.metadata.fps || 25.0;
  const durationSec = videoItem.metadata.duration_sec || 1.0;
  const totalFrames = videoItem.metadata.total_frames || Math.max(1, Math.round(durationSec * fps));

  // Fetch device info on mount
  useEffect(() => {
    getDevice().then(setDeviceInfo).catch(() => {});
  }, []);

  const handleDeviceChange = async (newPref: string) => {
    setDevicePref(newPref);
    try {
      const updated = await setDevicePreference(newPref);
      setDeviceInfo(updated);
    } catch {
      // quiet fail
    }
  };

  // Poll background job status
  useEffect(() => {
    if (!jobId) return;

    let isMounted = true;
    const interval = setInterval(async () => {
      try {
        const s = await getVideoStatus(jobId);
        if (isMounted) {
          setJobStatus(s);
          if (s.state === "completed" || s.state === "error" || s.state === "cancelled") {
            clearInterval(interval);
          }
        }
      } catch {
        // quiet poll fail
      }
    }, 1000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [jobId]);

  // Query frame annotations when currentFrameIdx changes
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    getVideoFrameAnnotations(jobId, currentFrameIdx)
      .then((res) => {
        if (!cancelled && res && res.tracks) {
          setCurrentTracks(res.tracks);
        }
      })
      .catch(() => {
        // quiet error on frame seek
      });

    return () => {
      cancelled = true;
    };
  }, [jobId, currentFrameIdx]);

  // Convert VideoTrackItem[] to CoralSegment[] so SegmentsTable & Sidebar render identical tables
  const segments: CoralSegment[] = useMemo(() => {
    return currentTracks.map((t) => ({
      id: t.track_id,
      id_str: `#${t.track_id}`,
      genus: t.genus,
      growth_form: t.growth_form,
      taxon_conf: t.taxon_conf,
      condition: t.condition,
      condition_conf: t.condition_conf,
      area_pct: roundAreaPct(t.area_px, videoItem.metadata.width, videoItem.metadata.height),
      area_px: t.area_px,
      predicted_iou: t.predicted_iou,
      color_hex: t.color_hex,
      centroid: t.centroid,
      bbox: t.bbox,
      polygon: t.polygon,
    }));
  }, [currentTracks, videoItem.metadata]);

  // Compute live SummaryStats for MetricCards & SegmentsTable
  const stats: SummaryStats = useMemo(() => {
    const totalPixels = Math.max(1, (videoItem.metadata.width || 972) * (videoItem.metadata.height || 720));
    const coveredPixels = segments.reduce((sum, s) => sum + s.area_px, 0);
    const coveragePct = Number(((coveredPixels / totalPixels) * 100).toFixed(2));
    const meanIou = segments.length > 0
      ? segments.reduce((sum, s) => sum + s.predicted_iou, 0) / segments.length
      : 0.85;

    return {
      total_corals_detected: segments.length,
      coral_coverage_pct: coveragePct,
      coral_covered_pixels: coveredPixels,
      total_image_pixels: totalPixels,
      image_resolution: `${videoItem.metadata.width}x${videoItem.metadata.height}`,
      mean_iou_confidence: Number(meanIou.toFixed(3)),
      mean_stability_score: 0.88,
    };
  }, [segments, videoItem.metadata]);

  // Handle Play/Pause
  const handleTogglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  // Handle Seek
  const handleSeekFrame = (targetFrame: number) => {
    if (!videoRef.current) return;
    const clampedFrame = Math.max(0, Math.min(totalFrames - 1, targetFrame));
    const targetSec = clampedFrame / fps;
    videoRef.current.currentTime = targetSec;
    setCurrentTimeSec(targetSec);
    setCurrentFrameIdx(clampedFrame);
  };

  // Step Frame
  const handleStepFrame = (dir: -1 | 1) => {
    handleSeekFrame(currentFrameIdx + dir);
  };

  // Time update from HTML5 video element
  const handleTimeUpdate = (timeSec: number) => {
    setCurrentTimeSec(timeSec);
    const frame = Math.min(totalFrames - 1, Math.floor(timeSec * fps));
    if (frame !== currentFrameIdx) {
      setCurrentFrameIdx(frame);
    }
  };

  const isJobProcessing = jobStatus ? jobStatus.state === "processing" : true;
  const keyframeIndices = jobStatus?.metadata ? (jobStatus as any).keyframe_indices || [0] : [0];

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-background text-foreground select-none">
      {/* TopBar matching Image Studio */}
      <TopBar
        onBack={onBackToUpload}
        deviceInfo={deviceInfo}
      />

      {/* Main Studio Body with exact same Sidebar + Main Content layout */}
      <div className="flex flex-1 overflow-hidden min-h-0 relative">
        {/* Collapsible Sidebar - EXACT same component! */}
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
          selectedMaskId={selectedTrackId}
          onSelectMaskId={setSelectedTrackId}
          deviceInfo={deviceInfo}
          devicePreference={devicePref}
          onDevicePreferenceChange={handleDeviceChange}
        />

        {/* Main Studio Canvas & Results with Natural Rhythm */}
        <main className="flex-1 min-w-0 h-full overflow-y-auto min-h-0 bg-muted/20">
          <div className="max-w-7xl mx-auto w-full p-6 lg:p-8 flex flex-col gap-6">
            {/* Video Navigation & Scrubber Bar (replaces PaginationBar with matching container & rhythm) */}
            <VideoTimelineBar
              currentFrame={currentFrameIdx}
              totalFrames={totalFrames}
              currentTimeSec={currentTimeSec}
              durationSec={durationSec}
              isPlaying={isPlaying}
              videoName={videoItem.video_name}
              keyframeIndices={keyframeIndices}
              onTogglePlay={handleTogglePlay}
              onStepFrame={handleStepFrame}
              onSeekFrame={handleSeekFrame}
            />

            {/* Canvas Video Viewer (built with exact same Card, header, badges, and layout modes as ImageViewer) */}
            <VideoViewer
              videoSrc={videoItem.previewUrl}
              layoutMode={layoutMode}
              colorMode={colorMode}
              alpha={alpha}
              drawContours={drawContours}
              drawLabels={drawLabels}
              drawBoxes={drawBoxes}
              segments={segments}
              selectedSegmentId={selectedTrackId}
              videoResolution={`${videoItem.metadata.width}x${videoItem.metadata.height}`}
              isLoading={isJobProcessing}
              currentTimeSec={currentTimeSec}
              isPlaying={isPlaying}
              onTimeUpdate={handleTimeUpdate}
              onVideoEnded={() => setIsPlaying(false)}
              videoRef={videoRef}
            />

            {/* 4 KPI Summary Cards - EXACT same component! */}
            <MetricCards stats={stats} isLoading={isJobProcessing} />

            {/* Detected Coral Segments Breakdown Table - EXACT same component! */}
            <SegmentsTable
              segments={segments}
              stats={stats}
              selectedSegmentId={selectedTrackId}
              onSelectSegment={setSelectedTrackId}
              isLoading={isJobProcessing}
            />

            {/* Export & Data Inspector - EXACT same Card and Accordion layout! */}
            <VideoExportPanel
              jobId={jobId}
              videoName={videoItem.video_name}
              hasResults={!isJobProcessing || segments.length > 0}
              rawJsonData={{
                job_id: jobId,
                current_frame: currentFrameIdx,
                timestamp_sec: currentTimeSec,
                active_corals_count: segments.length,
                stats: stats,
                current_tracks: segments,
              }}
            />
          </div>
        </main>
      </div>
    </div>
  );
};

function roundAreaPct(areaPx: number, width: number = 972, height: number = 720): number {
  const total = Math.max(1, width * height);
  return Number(((areaPx / total) * 100).toFixed(2));
}

export default VideoAnalysisView;
