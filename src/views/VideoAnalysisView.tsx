import React, { useState, useEffect, useRef } from "react";
import {
  ArrowLeft,
  Download,
  Activity,
  Layers,
  Cpu,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BrandLogo } from "../components/BrandLogo";
import { VideoControlsBar } from "../components/video/VideoControlsBar";
import { TransectMetricsChart } from "../components/video/TransectMetricsChart";
import { VideoExportModal } from "../components/video/VideoExportModal";
import {
  UploadedVideoItem,
  VideoJobStatus,
  VideoTrackItem,
  DeviceInfo,
} from "../types";
import {
  getVideoStatus,
  getVideoFrameAnnotations,
  getDevice,
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
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [jobId] = useState<string>(initialJobId || videoItem.job_id || "");
  const [jobStatus, setJobStatus] = useState<VideoJobStatus | null>(null);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);

  // Playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTimeSec, setCurrentTimeSec] = useState<number>(0);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);

  // Display Layer Controls
  const [showPolygons, setShowPolygons] = useState<boolean>(true);
  const [showBoxes, setShowBoxes] = useState<boolean>(false);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showHud, setShowHud] = useState<boolean>(true);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  // Active frame tracks
  const [currentTracks, setCurrentTracks] = useState<VideoTrackItem[]>([]);
  const [isExportModalOpen, setIsExportModalOpen] = useState<boolean>(false);

  const fps = videoItem.metadata.fps || 30.0;
  const durationSec = videoItem.metadata.duration_sec || 1.0;
  const totalFrames = videoItem.metadata.total_frames || Math.round(durationSec * fps);

  // Fetch device info on mount
  useEffect(() => {
    getDevice().then(setDeviceInfo).catch(() => {});
  }, []);

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
      } catch (err) {
        // quiet poll fail
      }
    }, 800);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [jobId]);

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
  const handleSeek = (timeSec: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, Math.min(durationSec, timeSec));
    setCurrentTimeSec(videoRef.current.currentTime);
  };

  // Handle 1-frame stepping
  const handleStepFrame = (direction: -1 | 1) => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    setIsPlaying(false);
    const stepSec = 1.0 / fps;
    handleSeek(videoRef.current.currentTime + direction * stepSec);
  };

  // Sync playback time & frame index
  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const curTime = videoRef.current.currentTime;
    setCurrentTimeSec(curTime);
    const frame = Math.min(totalFrames - 1, Math.floor(curTime * fps));
    setCurrentFrameIdx(frame);
  };

  // Change playback rate
  const handleChangePlaybackRate = (rate: number) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  };

  // Query frame annotations from backend or local map
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
        // If not ready yet, fallback to empty
      });

    return () => {
      cancelled = true;
    };
  }, [jobId, currentFrameIdx]);

  // Render Canvas Overlays synchronized with HTML5 video
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match canvas internal resolution to video display bounds
    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (currentTracks.length === 0) return;

    // Coordinate scale factors from original video dimensions to rendered canvas
    const scaleX = canvas.width / (videoItem.metadata.width || 1);
    const scaleY = canvas.height / (videoItem.metadata.height || 1);

    // 1. Draw Polygons
    if (showPolygons) {
      currentTracks.forEach((track) => {
        if (!track.polygon || track.polygon.length < 3) return;

        const isSelected = selectedTrackId === track.track_id;
        ctx.beginPath();
        const startX = track.polygon[0][0] * scaleX;
        const startY = track.polygon[0][1] * scaleY;
        ctx.moveTo(startX, startY);

        for (let i = 1; i < track.polygon.length; i++) {
          ctx.lineTo(track.polygon[i][0] * scaleX, track.polygon[i][1] * scaleY);
        }
        ctx.closePath();

        // Fill with semi-transparent color
        ctx.fillStyle = track.color_hex ? `${track.color_hex}44` : "rgba(6, 182, 212, 0.25)";
        ctx.fill();

        // Contour outline
        ctx.strokeStyle = isSelected ? "#ffffff" : track.color_hex || "#06b6d4";
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.stroke();
      });
    }

    // 2. Draw Bounding Boxes
    if (showBoxes) {
      currentTracks.forEach((track) => {
        const [x, y, bw, bh] = track.bbox;
        const isSelected = selectedTrackId === track.track_id;
        ctx.strokeStyle = isSelected ? "#ffffff" : track.color_hex || "#06b6d4";
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.strokeRect(x * scaleX, y * scaleY, bw * scaleX, bh * scaleY);
      });
    }

    // 3. Draw Badges & Labels
    if (showLabels) {
      ctx.font = "bold 11px system-ui, -apple-system, sans-serif";
      currentTracks.forEach((track) => {
        const [x, y] = track.bbox;
        const isBleached = track.is_bleached;
        const label = `#${track.track_id} ${track.genus} [${track.condition}]`;

        const renderX = Math.max(4, x * scaleX);
        const renderY = Math.max(16, y * scaleY - 6);

        const textWidth = ctx.measureText(label).width;

        // Pill background
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(renderX - 2, renderY - 12, textWidth + 8, 16);

        // Border
        ctx.strokeStyle = isBleached ? "#ef4444" : track.color_hex || "#06b6d4";
        ctx.lineWidth = 1;
        ctx.strokeRect(renderX - 2, renderY - 12, textWidth + 8, 16);

        // Text
        ctx.fillStyle = isBleached ? "#f87171" : "#f1f5f9";
        ctx.fillText(label, renderX + 2, renderY);
      });
    }
  }, [currentTracks, showPolygons, showBoxes, showLabels, selectedTrackId, videoItem.metadata]);

  // Statistics calculation for current view
  const activeCoralsCount = currentTracks.length;
  const bleachedCount = currentTracks.filter((t) => t.is_bleached).length;
  const bleachingPct = activeCoralsCount > 0 ? (bleachedCount / activeCoralsCount) * 100 : 0;
  const keyframeIndices = jobStatus?.metadata ? (jobStatus as any).keyframe_indices || [] : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Bar */}
      <header className="h-14 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-4 flex items-center justify-between shrink-0 z-20">
        <div className="flex items-center gap-3">
          <Button
            size="sm"
            variant="ghost"
            onClick={onBackToUpload}
            className="text-slate-400 hover:text-white gap-1.5 h-8 px-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Upload Studio</span>
          </Button>

          <div className="h-4 w-[1px] bg-slate-800" />

          <div className="flex items-center gap-2">
            <BrandLogo variant="topbar" />
            <span className="font-bold text-sm tracking-tight text-white">ReefVision</span>
            <Badge variant="outline" className="text-[10px] bg-cyan-500/10 text-cyan-400 border-cyan-500/30">
              Live Video Transect Studio
            </Badge>
          </div>
        </div>

        {/* Video metadata & Hardware status */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 text-xs text-slate-400 font-mono">
            <span>{videoItem.video_name}</span>
            <span>•</span>
            <span>
              {videoItem.metadata.width}x{videoItem.metadata.height}
            </span>
            <span>•</span>
            <span>{videoItem.metadata.fps} FPS</span>
          </div>

          <div className="h-4 w-[1px] bg-slate-800" />

          {/* Hardware acceleration badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[11px] font-medium">{deviceInfo?.gpu_name || "Hardware Engine"}</span>
          </div>

          <Button
            size="sm"
            onClick={() => setIsExportModalOpen(true)}
            className="bg-cyan-500 text-slate-950 hover:bg-cyan-400 font-semibold h-8 gap-1.5"
          >
            <Download className="w-3.5 h-3.5" />
            Export Results
          </Button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left / Center: Video Player & Overlays */}
        <main className="flex-1 flex flex-col p-4 gap-3 overflow-y-auto">
          {/* Main Video Stage */}
          <div className="relative w-full bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center min-h-[460px]">
            {/* HTML5 Video Element */}
            <video
              ref={videoRef}
              src={videoItem.previewUrl}
              onTimeUpdate={handleTimeUpdate}
              onEnded={() => setIsPlaying(false)}
              className="max-h-[68vh] w-auto mx-auto object-contain select-none"
              playsInline
            />

            {/* Overlaid Vector Canvas */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none select-none"
            />

            {/* Top HUD Banner on Video */}
            {showHud && (
              <div className="absolute top-3 left-3 right-3 flex items-center justify-between px-3 py-1.5 rounded-lg bg-slate-950/80 border border-slate-800/80 backdrop-blur-md text-xs font-mono text-slate-300 pointer-events-none">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                    <Activity className="w-3.5 h-3.5" /> Transect Tracker
                  </span>
                  <span>Frame {currentFrameIdx + 1}/{totalFrames}</span>
                  <span>Time: {currentTimeSec.toFixed(1)}s</span>
                </div>
                <div className="flex items-center gap-4">
                  <span>
                    Corals in View: <b className="text-white">{activeCoralsCount}</b>
                  </span>
                  <span>
                    Bleaching:{" "}
                    <b className={bleachingPct > 20 ? "text-rose-400" : "text-emerald-400"}>
                      {bleachingPct.toFixed(1)}%
                    </b>
                  </span>
                  <span className="text-cyan-400">
                    {(jobStatus?.fps_realtime || fps).toFixed(1)} FPS
                  </span>
                </div>
              </div>
            )}

            {/* Layer Visibility Toggles Floating Tool */}
            <div className="absolute bottom-3 right-3 flex items-center gap-1.5 bg-slate-900/90 border border-slate-800 p-1.5 rounded-lg shadow-lg backdrop-blur-md">
              <Button
                size="sm"
                variant={showPolygons ? "default" : "ghost"}
                onClick={() => setShowPolygons(!showPolygons)}
                title="Toggle Coral Polygons"
                className={`h-7 px-2 text-xs gap-1 ${
                  showPolygons ? "bg-cyan-500 text-slate-950 font-medium" : "text-slate-400"
                }`}
              >
                <Layers className="w-3.5 h-3.5" /> Polygons
              </Button>
              <Button
                size="sm"
                variant={showBoxes ? "default" : "ghost"}
                onClick={() => setShowBoxes(!showBoxes)}
                title="Toggle Bounding Boxes"
                className={`h-7 px-2 text-xs gap-1 ${
                  showBoxes ? "bg-cyan-500 text-slate-950 font-medium" : "text-slate-400"
                }`}
              >
                BBoxes
              </Button>
              <Button
                size="sm"
                variant={showLabels ? "default" : "ghost"}
                onClick={() => setShowLabels(!showLabels)}
                title="Toggle Species Labels"
                className={`h-7 px-2 text-xs gap-1 ${
                  showLabels ? "bg-cyan-500 text-slate-950 font-medium" : "text-slate-400"
                }`}
              >
                Labels
              </Button>
              <Button
                size="sm"
                variant={showHud ? "default" : "ghost"}
                onClick={() => setShowHud(!showHud)}
                title="Toggle Benthic HUD"
                className={`h-7 px-2 text-xs gap-1 ${
                  showHud ? "bg-cyan-500 text-slate-950 font-medium" : "text-slate-400"
                }`}
              >
                HUD
              </Button>
            </div>
          </div>

          {/* Video Playback Controls Bar */}
          <VideoControlsBar
            isPlaying={isPlaying}
            onTogglePlay={handleTogglePlay}
            onSeek={handleSeek}
            onStepFrame={handleStepFrame}
            currentTime={currentTimeSec}
            duration={durationSec}
            currentFrame={currentFrameIdx}
            totalFrames={totalFrames}
            fps={fps}
            realtimeFps={jobStatus?.fps_realtime || fps}
            keyframeIndices={keyframeIndices}
            playbackRate={playbackRate}
            onChangePlaybackRate={handleChangePlaybackRate}
          />

          {/* Live Transect Time-Series Chart */}
          <TransectMetricsChart
            timeSeries={jobStatus?.time_series || []}
            currentTimeSec={currentTimeSec}
            durationSec={durationSec}
            onSeek={handleSeek}
          />
        </main>

        {/* Right Sidebar: Transect Telemetry & Tracked Corals Table */}
        <aside className="w-80 border-l border-slate-800 bg-slate-900/50 p-4 flex flex-col gap-4 overflow-y-auto shrink-0">
          <div>
            <h3 className="text-sm font-semibold text-slate-200 mb-1">Transect Summary</h3>
            <p className="text-xs text-slate-500">Autonomous benthic colony monitoring</p>
          </div>

          {/* Telemetry KPI Cards */}
          <div className="grid grid-cols-2 gap-2.5">
            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">
                In View
              </span>
              <span className="text-xl font-bold text-cyan-400 font-mono">
                {activeCoralsCount}
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">
                Total Tracked
              </span>
              <span className="text-xl font-bold text-white font-mono">
                {jobStatus?.total_unique_corals || activeCoralsCount}
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">
                Bleaching %
              </span>
              <span
                className={`text-xl font-bold font-mono ${
                  bleachingPct > 20 ? "text-rose-400" : "text-emerald-400"
                }`}
              >
                {bleachingPct.toFixed(1)}%
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">
                Keyframes
              </span>
              <span className="text-xl font-bold text-amber-400 font-mono">
                {jobStatus?.keyframes_count || 1}
              </span>
            </div>
          </div>

          {/* Active Corals in View Table */}
          <div className="flex-1 flex flex-col min-h-0 bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden">
            <div className="p-3 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">
                Active Colonies ({currentTracks.length})
              </span>
              <span className="text-[11px] text-slate-500 font-mono">Frame {currentFrameIdx + 1}</span>
            </div>

            <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60 text-xs">
              {currentTracks.length === 0 ? (
                <div className="p-4 text-center text-slate-500 text-xs">
                  No coral colonies detected in current frame.
                </div>
              ) : (
                currentTracks.map((track) => {
                  const isSelected = selectedTrackId === track.track_id;
                  const isBleached = track.is_bleached;
                  return (
                    <div
                      key={track.track_id}
                      onClick={() =>
                        setSelectedTrackId(isSelected ? null : track.track_id)
                      }
                      className={`p-2.5 flex items-center justify-between cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-cyan-500/15 border-l-2 border-cyan-400"
                          : "hover:bg-slate-800/40"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full shrink-0"
                          style={{ backgroundColor: track.color_hex }}
                        />
                        <div>
                          <div className="font-medium text-slate-200 flex items-center gap-1.5">
                            <span>#{track.track_id}</span>
                            <span>{track.genus}</span>
                          </div>
                          <span className="text-[10px] text-slate-400">
                            {track.growth_form} • {track.area_pct}% coverage
                          </span>
                        </div>
                      </div>

                      <Badge
                        variant="outline"
                        className={`text-[10px] ${
                          isBleached
                            ? "bg-rose-500/10 text-rose-400 border-rose-500/30"
                            : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                        }`}
                      >
                        {track.condition}
                      </Badge>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* Export Results Modal */}
      <VideoExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        jobStatus={jobStatus}
      />
    </div>
  );
};
