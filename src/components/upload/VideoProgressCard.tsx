import React from "react";
import {
  Rocket,
  CheckCircle2,
  Play,
  Pause,
  StopCircle,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { VideoJobStatus, UploadedVideoItem } from "../../types";

export interface VideoProgressCardProps {
  videoJobStatus: VideoJobStatus | null;
  uploadedVideo: UploadedVideoItem | null;
  isStartingVideoJob: boolean;
  isUploadingVideo: boolean;
  isBackendConnected: boolean;
  isPausingResumingVideo: boolean;
  onStartVideoAnalysis: () => void;
  onPauseResumeVideo: () => void;
  onCancelVideo: () => void;
  onOpenVideoStudioFromPanel: () => void;
}

export const VideoProgressCard: React.FC<VideoProgressCardProps> = ({
  videoJobStatus,
  uploadedVideo,
  isStartingVideoJob,
  isUploadingVideo,
  isBackendConnected,
  isPausingResumingVideo,
  onStartVideoAnalysis,
  onPauseResumeVideo,
  onCancelVideo,
  onOpenVideoStudioFromPanel,
}) => {
  const jobState = videoJobStatus?.state;
  // Starting/cancelling jobs are still live work, so the card and its Cancel button stay visible
  const isStartingOrCancelling = jobState === "queued" || jobState === "preparing" || jobState === "cancelling";
  const isCancelling = jobState === "cancelling";
  const isRunning = jobState === "processing" || isStartingOrCancelling;
  const isPaused = videoJobStatus?.state === "paused";
  const isComplete = videoJobStatus?.state === "completed";
  const hasProcessed = (videoJobStatus?.current_frame ?? 0) > 0;

  if (!isRunning && !isPaused && !isComplete && !hasProcessed) {
    return (
      <div className="shrink-0 pt-2.5 mt-auto border-t border-slate-100 bg-white">
        <Button
          type="button"
          className="w-full h-10 text-xs font-bold tracking-wider uppercase text-white rounded-xl shadow-xs hover:shadow-md transition-all gap-2 bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] hover:opacity-95"
          onClick={onStartVideoAnalysis}
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
                {!isStartingOrCancelling && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={onPauseResumeVideo}
                  disabled={isPausingResumingVideo}
                  className="h-7 px-2 text-[10px] font-semibold gap-1 text-slate-700 hover:bg-slate-100"
                  title={isPaused ? "Resume processing" : "Pause processing"}
                >
                  {isPaused ? <Play size={11} /> : <Pause size={11} />}
                  <span>{isPaused ? "Resume" : "Pause"}</span>
                </Button>
                )}

                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={onCancelVideo}
                  disabled={isCancelling}
                  className="h-7 px-2 text-[10px] font-semibold gap-1 text-red-600 border-red-200 hover:bg-red-50"
                  title="Cancel video tracking"
                >
                  <StopCircle size={11} />
                  <span>{isCancelling ? "Cancelling…" : "Cancel"}</span>
                </Button>
              </>
            )}

            {isComplete && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={onStartVideoAnalysis}
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
          indicatorColor={
            isRunning
              ? "bg-gradient-to-r from-cyan-600 to-teal-500"
              : isPaused
              ? "bg-amber-500"
              : "bg-emerald-600"
          }
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
            onClick={onOpenVideoStudioFromPanel}
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
