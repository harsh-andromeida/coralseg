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
  AlertTriangle,
  Activity,
  Zap,
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
  getVideoFramesDelta,
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
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const frameCacheRef = useRef<Map<number, VideoTrackItem[]>>(new Map());
  const activeFrameRef = useRef<number>(0);
  const lastCommittedSeqRef = useRef<number>(0);
  const isPollingDeltaRef = useRef<boolean>(false);
  const [isUserScrolledUp, setIsUserScrolledUp] = useState<boolean>(false);

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

  const isJobProcessing = jobStatus ? jobStatus.state === "processing" : true;
  const isProcessingState = jobStatus?.state === "processing" || jobStatus?.state === "paused";
  const maxProcessedFrame = isProcessingState
    ? Math.max(0, (jobStatus?.current_frame || 1) - 1)
    : Math.max(0, totalFrames - 1);
  const maxNavigableFrame = isProcessingState ? maxProcessedFrame : Math.max(0, totalFrames - 1);

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

  // Poll background job status, drain delta pages, and advance live video stream
  useEffect(() => {
    if (!jobId) return;

    let isMounted = true;

    const pollAndDrain = async () => {
      if (isPollingDeltaRef.current) return;
      isPollingDeltaRef.current = true;

      try {
        const s = await getVideoStatus(jobId);
        if (!isMounted) return;
        setJobStatus(s);

        // Drain delta pages until caught up or batch limit reached (supports completed jobs)
        let hasMore = true;
        let drainCycles = 0;
        while (hasMore && drainCycles < 4 && isMounted) {
          drainCycles++;
          const deltaRes = await getVideoFramesDelta(jobId, lastCommittedSeqRef.current, 64);
          if (!isMounted) break;

          if (deltaRes && deltaRes.frames && deltaRes.frames.length > 0) {
            deltaRes.frames.forEach((item) => {
              frameCacheRef.current.set(item.frame_idx, item.tracks);
            });
            if (deltaRes.max_seq > lastCommittedSeqRef.current) {
              lastCommittedSeqRef.current = deltaRes.max_seq;
            }
            if (deltaRes.frames.length < 64) {
              hasMore = false;
            }
          } else {
            hasMore = false;
          }
        }

        // Bound in-memory cache to prevent memory bloat on large videos
        if (frameCacheRef.current.size > 800) {
          const active = activeFrameRef.current;
          const sortedByDist = Array.from(frameCacheRef.current.keys()).sort(
            (a, b) => Math.abs(b - active) - Math.abs(a - active)
          );
          for (let i = 0; i < 200 && i < sortedByDist.length; i++) {
            frameCacheRef.current.delete(sortedByDist[i]);
          }
        }

        if (frameCacheRef.current.has(activeFrameRef.current)) {
          setCurrentTracks(frameCacheRef.current.get(activeFrameRef.current)!);
        }

        // Auto-advance player to follow live stream if user is following live
        if (s.state === "processing" && isFollowingLive && !isPlaying && s.current_frame > 0) {
          const liveFrame = Math.max(0, Math.min(totalFrames - 1, s.current_frame - 1));
          activeFrameRef.current = liveFrame;
          setCurrentFrameIdx(liveFrame);
          const liveSec = liveFrame / fps;
          setCurrentTimeSec(liveSec);
          if (videoRef.current) {
            videoRef.current.currentTime = liveSec;
          }
          if (frameCacheRef.current.has(liveFrame)) {
            setCurrentTracks(frameCacheRef.current.get(liveFrame)!);
          }
        }

        // Only clear interval when job is in terminal state AND all pages have been drained
        const isJobTerminal = s.state === "completed" || s.state === "error" || s.state === "cancelled";
        const isCaughtUp = !hasMore && (lastCommittedSeqRef.current >= (s.total_frames || 0) || s.current_frame === 0);
        if (isJobTerminal && isCaughtUp) {
          clearInterval(interval);
        }
      } catch {
        // quiet poll fail
      } finally {
        isPollingDeltaRef.current = false;
      }
    };

    // Execute immediately on mount to drain initial frames
    pollAndDrain();
    const interval = setInterval(pollAndDrain, 400);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [jobId, isFollowingLive, isPlaying, totalFrames, fps]);

  // Query frame annotations when currentFrameIdx changes (instant in-memory cache lookup)
  useEffect(() => {
    if (!jobId) return;
    activeFrameRef.current = currentFrameIdx;

    // Instant return if already in client cache
    if (frameCacheRef.current.has(currentFrameIdx)) {
      setCurrentTracks(frameCacheRef.current.get(currentFrameIdx)!);
      return;
    }

    const targetFrame = currentFrameIdx;
    getVideoFrameAnnotations(jobId, targetFrame)
      .then((res) => {
        if (res && res.tracks) {
          // Always cache the retrieved frame tracks so future scrubbing is instant
          frameCacheRef.current.set(targetFrame, res.tracks);
          if (activeFrameRef.current === targetFrame) {
            setCurrentTracks(res.tracks);
          }
        }
      })
      .catch(() => {
        // quiet error on frame seek
      });
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
      // If at or beyond max processed frame, loop back to frame 0 so user can watch from start
      if (maxNavigableFrame > 0 && currentFrameIdx >= maxNavigableFrame) {
        handleSeekFrame(0);
      }
      setIsFollowingLive(false);
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  // High-FPS playback sync: keeps tracking overlays locked to video frames at 60 FPS
  useEffect(() => {
    if (!isPlaying) return;
    let animId: number;

    const syncPlayback = () => {
      const vid = videoRef.current;
      if (vid && !vid.paused) {
        const time = vid.currentTime;
        const maxSec = maxNavigableFrame / fps;
        if (isProcessingState && time >= maxSec) {
          vid.pause();
          setIsPlaying(false);
          setCurrentTimeSec(maxSec);
          setCurrentFrameIdx(maxNavigableFrame);
          if (frameCacheRef.current.has(maxNavigableFrame)) {
            setCurrentTracks(frameCacheRef.current.get(maxNavigableFrame)!);
          }
          return;
        }

        const f = Math.min(maxNavigableFrame, Math.floor(time * fps));
        setCurrentTimeSec(time);
        if (f !== activeFrameRef.current) {
          activeFrameRef.current = f;
          setCurrentFrameIdx(f);
          if (frameCacheRef.current.has(f)) {
            setCurrentTracks(frameCacheRef.current.get(f)!);
          }
        }
      }
      animId = requestAnimationFrame(syncPlayback);
    };

    animId = requestAnimationFrame(syncPlayback);
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, maxNavigableFrame, fps, isProcessingState]);

  // Handle Seek
  const handleSeekFrame = (targetFrame: number) => {
    if (!videoRef.current) return;
    const clampedFrame = Math.max(0, Math.min(maxNavigableFrame, targetFrame));
    const targetSec = clampedFrame / fps;
    videoRef.current.currentTime = targetSec;
    setCurrentTimeSec(targetSec);
    activeFrameRef.current = clampedFrame;
    setCurrentFrameIdx(clampedFrame);

    if (frameCacheRef.current.has(clampedFrame)) {
      setCurrentTracks(frameCacheRef.current.get(clampedFrame)!);
    }

    // If seeking to the latest processed frame, resume following live stream
    if (isProcessingState && clampedFrame >= maxNavigableFrame) {
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
    const maxAllowedSec = maxNavigableFrame / fps;

    if (isProcessingState && timeSec >= maxAllowedSec) {
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.currentTime = maxAllowedSec;
      }
      setIsPlaying(false);
      setCurrentTimeSec(maxAllowedSec);
      activeFrameRef.current = maxNavigableFrame;
      setCurrentFrameIdx(maxNavigableFrame);
      if (frameCacheRef.current.has(maxNavigableFrame)) {
        setCurrentTracks(frameCacheRef.current.get(maxNavigableFrame)!);
      }
      return;
    }

    setCurrentTimeSec(timeSec);
    const frame = Math.min(maxNavigableFrame, Math.floor(timeSec * fps));
    if (frame !== activeFrameRef.current) {
      activeFrameRef.current = frame;
      setCurrentFrameIdx(frame);
      if (frameCacheRef.current.has(frame)) {
        setCurrentTracks(frameCacheRef.current.get(frame)!);
      }
    }
  };

  // Activity log scroll handler: detects if user has scrolled up to inspect previous logs
  const handleLogsScroll = () => {
    if (!logsContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logsContainerRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    setIsUserScrolledUp(distanceFromBottom > 35);
  };

  // Safe auto-scroll: scrolls ONLY the internal container and only when user hasn't scrolled up
  useEffect(() => {
    if (showLogs && !isUserScrolledUp && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [jobStatus?.logs, showLogs, isUserScrolledUp]);

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
              maxProcessedFrame={maxProcessedFrame}
              isProcessing={isProcessingState}
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
                    ) : jobStatus.state === "error" ? (
                      <AlertTriangle size={16} className="text-rose-600 shrink-0" />
                    ) : (
                      <CheckCircle2 size={16} className="text-emerald-600 shrink-0" />
                    )}

                    <div className="flex items-center gap-2 truncate">
                      <span className={`text-xs font-bold tracking-tight ${jobStatus.state === "error" ? "text-rose-700" : "text-slate-900"}`}>
                        {jobStatus.state === "processing"
                          ? "Autonomous Pipeline Active"
                          : jobStatus.state === "paused"
                          ? "Pipeline Paused"
                          : jobStatus.state === "error"
                          ? "Pipeline Execution Error"
                          : "Transect Processing Complete"}
                      </span>

                      <span className={`text-[11px] font-mono px-2 py-0.5 rounded-md font-bold ${
                        jobStatus.state === "error"
                          ? "bg-rose-50 border border-rose-200 text-rose-900"
                          : "bg-cyan-50 border border-cyan-200 text-cyan-900"
                      }`}>
                        Frame {jobStatus.current_frame} / {totalFrames} ({jobStatus.progress_pct}%)
                      </span>

                      {((jobStatus.analysis_fps ?? jobStatus.fps_realtime) > 0) && (
                        <span className="text-[10px] font-mono font-bold text-cyan-800 bg-cyan-100/70 border border-cyan-200/60 px-1.5 py-0.5 rounded" title="Total pipeline source throughput">
                          {(jobStatus.analysis_fps ?? jobStatus.fps_realtime).toFixed(1)} Source FPS
                        </span>
                      )}

                      {jobStatus.tracker_fps && jobStatus.tracker_fps > 0 ? (
                        <span className="text-[10px] font-mono font-bold text-teal-800 bg-teal-100/70 border border-teal-200/60 px-1.5 py-0.5 rounded" title="Pure SAM tracker inference throughput">
                          {jobStatus.tracker_fps.toFixed(1)} Tracker FPS
                        </span>
                      ) : null}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {/* Device & Precision Pill */}
                    {deviceInfo && (
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 text-white text-[11px] font-semibold shadow-xs" title={`Accelerator: ${deviceInfo.device_name || deviceInfo.gpu_name || deviceInfo.device}`}>
                        <Zap size={12} className="text-cyan-400" />
                        <span className="text-slate-300 font-mono uppercase">{deviceInfo.device}</span>
                        {deviceInfo.native_precision && (
                          <span className="text-cyan-300 font-mono text-[10px]">
                            {deviceInfo.native_precision.toUpperCase()}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Active Model Pill */}
                    {jobStatus.current_model && (
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 text-white text-[11px] font-semibold shadow-xs">
                        <Cpu size={12} className="text-cyan-400" />
                        <span className="text-cyan-300 font-bold">{jobStatus.current_model}</span>
                      </div>
                    )}

                    {/* Export Status Pill */}
                    {jobStatus.export_status && (
                      <div className={`flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-bold border ${
                        jobStatus.export_status === "ready"
                          ? "bg-emerald-50 border-emerald-300 text-emerald-800"
                          : jobStatus.export_status === "pending"
                          ? "bg-amber-50 border-amber-300 text-amber-800 animate-pulse"
                          : "bg-rose-50 border-rose-300 text-rose-800"
                      }`} title="Background export state">
                        <span>EXPORT: {jobStatus.export_status.toUpperCase()}</span>
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
                        : jobStatus.state === "error"
                        ? "bg-rose-500"
                        : "bg-gradient-to-r from-cyan-600 via-teal-500 to-cyan-400"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, jobStatus.progress_pct))}%` }}
                  />
                </div>

                {/* Explicit Error Diagnostic Display */}
                {jobStatus.state === "error" && jobStatus.error && (
                  <div className="rounded-xl bg-rose-50 border border-rose-200/80 p-3 text-xs text-rose-800 flex items-start gap-2.5">
                    <AlertTriangle size={15} className="text-rose-600 shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold text-rose-900">Pipeline Execution Error</p>
                      <p className="text-rose-700 font-mono mt-0.5 text-[11px]">{jobStatus.error}</p>
                    </div>
                  </div>
                )}

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
                  <div
                    ref={logsContainerRef}
                    onScroll={handleLogsScroll}
                    className="mt-2 rounded-xl bg-slate-950 border border-slate-800 p-3 font-mono text-[11px] text-slate-300 space-y-1 max-h-36 overflow-y-auto portal-scroll-area shadow-inner relative"
                  >
                    <div className="sticky top-0 bg-slate-950/95 backdrop-blur-xs flex items-center justify-between text-[10px] text-slate-500 border-b border-slate-800 pb-1.5 mb-1.5 z-10">
                      <span className="flex items-center gap-1 font-bold tracking-wider uppercase text-cyan-400">
                        <Activity size={11} /> Real-time Pipeline Telemetry & Model Logs
                      </span>
                      <div className="flex items-center gap-2">
                        {isUserScrolledUp ? (
                          <button
                            type="button"
                            onClick={() => {
                              setIsUserScrolledUp(false);
                              if (logsContainerRef.current) {
                                logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
                              }
                            }}
                            className="text-[10px] text-cyan-300 bg-cyan-950 border border-cyan-700/80 px-2 py-0.5 rounded hover:bg-cyan-900 transition-colors flex items-center gap-1 font-sans cursor-pointer"
                          >
                            <span>Resume auto-scroll</span>
                            <span>↓</span>
                          </button>
                        ) : (
                          <span>Engine: {jobStatus.tracker_mode === "efficienttam" ? "EfficientTAM (experimental)" : "SAM 2.1 Video Predictor"}</span>
                        )}
                      </div>
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
              exportPaths={jobStatus?.export_paths as any}
              exportStatus={jobStatus?.export_status}
              isProcessing={isProcessingState}
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
