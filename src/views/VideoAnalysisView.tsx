import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  Cpu,
  Terminal,
  ChevronDown,
  ChevronUp,
  Play,
  Pause,
  StopCircle,
  CheckCircle2,
  Activity,
} from "lucide-react";
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
  pauseVideoJob,
  resumeVideoJob,
  cancelVideoJob,
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
  const logsEndRef = useRef<HTMLDivElement>(null);

  const [jobId] = useState<string>(initialJobId || videoItem.job_id || "");
  const [jobStatus, setJobStatus] = useState<VideoJobStatus | null>(null);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [devicePref, setDevicePref] = useState<string>("auto");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);
  const [showLogs, setShowLogs] = useState<boolean>(true);
  const [isTogglingPause, setIsTogglingPause] = useState<boolean>(false);

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
  const [isFollowingLive, setIsFollowingLive] = useState<boolean>(true);

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

  // Poll background job status and advance live video stream
  useEffect(() => {
    if (!jobId) return;

    let isMounted = true;
    const interval = setInterval(async () => {
      try {
        const s = await getVideoStatus(jobId);
        if (isMounted) {
          setJobStatus(s);

          // Auto-advance player to follow live stream as frames are completed
          if (s.state === "processing" && isFollowingLive && s.current_frame > 0) {
            const liveFrame = Math.max(0, Math.min(totalFrames - 1, s.current_frame - 1));
            setCurrentFrameIdx(liveFrame);
            const liveSec = liveFrame / fps;
            setCurrentTimeSec(liveSec);
            if (videoRef.current) {
              videoRef.current.currentTime = liveSec;
            }
          }

          if (s.state === "completed" || s.state === "error" || s.state === "cancelled") {
            clearInterval(interval);
          }
        }
      } catch {
        // quiet poll fail
      }
    }, 400);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [jobId, isFollowingLive, totalFrames, fps]);

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

    // If seeking to the latest processed frame, resume following live stream
    if (jobStatus && clampedFrame >= Math.max(0, jobStatus.current_frame - 1)) {
      setIsFollowingLive(true);
    } else {
      setIsFollowingLive(false);
    }
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

  // Auto-scroll activity log to bottom
  useEffect(() => {
    if (showLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [jobStatus?.logs, showLogs]);

  const handleTogglePauseJob = async () => {
    if (!jobId || !jobStatus) return;
    setIsTogglingPause(true);
    try {
      if (jobStatus.state === "paused") {
        await resumeVideoJob(jobId);
      } else {
        await pauseVideoJob(jobId);
      }
      const updated = await getVideoStatus(jobId);
      setJobStatus(updated);
    } catch (err: any) {
      alert(`Error toggling pause: ${err?.message || err}`);
    } finally {
      setIsTogglingPause(false);
    }
  };

  const handleCancelJob = async () => {
    if (!jobId) return;
    try {
      await cancelVideoJob(jobId);
      const updated = await getVideoStatus(jobId);
      setJobStatus(updated);
    } catch (err: any) {
      alert(`Error cancelling video: ${err?.message || err}`);
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

            {/* Live Pipeline Telemetry HUD & Model Activity Inspector */}
            {jobStatus && (
              <div className="rounded-2xl border border-cyan-200/90 bg-white/95 backdrop-blur-md shadow-xs p-4 space-y-3">
                {/* Header: State, Frame counter, Realtime FPS, Active Model, Controls */}
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5 min-w-0">
                    {jobStatus.state === "processing" ? (
                      <span className="relative flex h-3 w-3 shrink-0">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-600" />
                      </span>
                    ) : jobStatus.state === "paused" ? (
                      <span className="h-3 w-3 rounded-full bg-amber-500 shrink-0" />
                    ) : (
                      <CheckCircle2 size={16} className="text-emerald-600 shrink-0" />
                    )}

                    <div className="flex items-center gap-2 truncate">
                      <span className="text-xs font-bold text-slate-900 tracking-tight">
                        {jobStatus.state === "processing"
                          ? "Autonomous Pipeline Active"
                          : jobStatus.state === "paused"
                          ? "Pipeline Paused"
                          : "Transect Processing Complete"}
                      </span>

                      <span className="text-[11px] font-mono px-2 py-0.5 rounded-md bg-cyan-50 border border-cyan-200 text-cyan-900 font-bold">
                        Frame {jobStatus.current_frame} / {totalFrames} ({jobStatus.progress_pct}%)
                      </span>

                      {jobStatus.fps_realtime > 0 && (
                        <span className="text-[10px] font-mono font-bold text-cyan-800 bg-cyan-100/70 border border-cyan-200/60 px-1.5 py-0.5 rounded">
                          {jobStatus.fps_realtime} FPS
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {/* Active Model Pill */}
                    {jobStatus.current_model && (
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 text-white text-[11px] font-semibold shadow-xs">
                        <Cpu size={12} className="text-cyan-400" />
                        <span className="text-slate-300">Model:</span>
                        <span className="text-cyan-300 font-bold">{jobStatus.current_model}</span>
                      </div>
                    )}

                    {/* Pause / Resume & Cancel Buttons */}
                    {(jobStatus.state === "processing" || jobStatus.state === "paused") && (
                      <>
                        <button
                          type="button"
                          onClick={handleTogglePauseJob}
                          disabled={isTogglingPause}
                          className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 shadow-2xs transition-all"
                          title={jobStatus.state === "paused" ? "Resume analysis" : "Pause analysis"}
                        >
                          {jobStatus.state === "paused" ? <Play size={12} /> : <Pause size={12} />}
                          <span>{jobStatus.state === "paused" ? "Resume" : "Pause"}</span>
                        </button>

                        <button
                          type="button"
                          onClick={handleCancelJob}
                          className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold border border-red-200 bg-red-50/50 hover:bg-red-100/60 text-red-700 shadow-2xs transition-all"
                          title="Cancel analysis"
                        >
                          <StopCircle size={12} />
                          <span>Cancel</span>
                        </button>
                      </>
                    )}

                    {/* Toggle Activity Log Stream */}
                    <button
                      type="button"
                      onClick={() => setShowLogs(!showLogs)}
                      className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all ${
                        showLogs
                          ? "bg-slate-100 border-slate-300 text-slate-900"
                          : "bg-white border-slate-200 text-slate-600 hover:text-slate-900"
                      }`}
                      title="Toggle live model log stream"
                    >
                      <Terminal size={12} className={showLogs ? "text-cyan-600" : ""} />
                      <span>Activity Log</span>
                      <span className="text-[10px] font-mono px-1 rounded bg-slate-200/60 text-slate-700">
                        {jobStatus.logs?.length || 0}
                      </span>
                      {showLogs ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </button>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200/60">
                  <div
                    className={`h-full transition-all duration-300 ${
                      jobStatus.state === "completed"
                        ? "bg-emerald-500"
                        : jobStatus.state === "paused"
                        ? "bg-amber-500"
                        : "bg-gradient-to-r from-cyan-600 via-teal-500 to-cyan-400"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, jobStatus.progress_pct))}%` }}
                  />
                </div>

                {/* Sub-bar: Stage description & live colonies census */}
                <div className="flex flex-wrap items-center justify-between text-xs text-slate-600 gap-2 pt-0.5">
                  <div className="flex items-center gap-2 truncate">
                    <span className="font-semibold text-slate-800">
                      {jobStatus.current_stage || "Analyzing benthic habitat..."}
                    </span>
                    {jobStatus.stage_detail && (
                      <span className="text-slate-400 truncate text-[11px]">
                        • {jobStatus.stage_detail}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-3 text-[11px] font-medium shrink-0">
                    <span className="text-slate-500">
                      Discovered:{" "}
                      <strong className="text-cyan-800 font-bold">{jobStatus.total_unique_corals || 0}</strong> colonies
                    </span>
                    <span className="text-slate-300">•</span>
                    <span className="text-slate-500">
                      Keyframes:{" "}
                      <strong className="text-slate-800 font-bold">{jobStatus.keyframes_count || 0}</strong>
                    </span>
                    <span className="text-slate-300">•</span>
                    <span className="text-slate-500">
                      Bleaching:{" "}
                      <strong className="text-amber-700 font-bold">{jobStatus.bleaching_prevalence_pct || 0}%</strong>
                    </span>
                  </div>
                </div>

                {/* Collapsible Activity Log Console */}
                {showLogs && (
                  <div className="mt-2 rounded-xl bg-slate-950 border border-slate-800 p-3 font-mono text-[11px] text-slate-300 space-y-1 max-h-36 overflow-y-auto portal-scroll-area shadow-inner">
                    <div className="flex items-center justify-between text-[10px] text-slate-500 border-b border-slate-800 pb-1.5 mb-1.5">
                      <span className="flex items-center gap-1 font-bold tracking-wider uppercase text-cyan-400">
                        <Activity size={11} /> Real-time Pipeline Telemetry & Model Logs
                      </span>
                      <span>Engine: {jobStatus.tracker_mode === "sam2" ? "SAM 2 Video Predictor" : "Lucas-Kanade Optical Flow (60-120 FPS)"}</span>
                    </div>

                    {jobStatus.logs && jobStatus.logs.length > 0 ? (
                      jobStatus.logs.map((logStr, lIdx) => (
                        <div key={lIdx} className="leading-relaxed flex items-start gap-2">
                          <span className="text-slate-500 shrink-0 select-none">›</span>
                          <span className={logStr.includes("Keyframe") ? "text-cyan-300 font-semibold" : logStr.includes("BioCLIP") ? "text-emerald-300" : logStr.includes("Completed") || logStr.includes("finished") ? "text-amber-300 font-bold" : "text-slate-300"}>
                            {logStr}
                          </span>
                        </div>
                      ))
                    ) : (
                      <div className="text-slate-500 italic">No events logged yet. Waiting for pipeline start...</div>
                    )}
                    <div ref={logsEndRef} />
                  </div>
                )}
              </div>
            )}

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

            {/* 4 KPI Summary Cards - Live Progressive Census */}
            <MetricCards stats={stats} isLoading={!jobStatus} />

            {/* Detected Coral Segments Breakdown Table - Live Updating */}
            <SegmentsTable
              segments={segments}
              stats={stats}
              selectedSegmentId={selectedTrackId}
              onSelectSegment={setSelectedTrackId}
              isLoading={!jobStatus}
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
