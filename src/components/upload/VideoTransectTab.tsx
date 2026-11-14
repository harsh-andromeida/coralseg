import React, { useState, useRef } from "react";
import { Sparkles, AlertCircle, Film, Video, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { UploadedVideoItem, VideoJobStatus } from "../../types";
import { VideoProgressCard } from "./VideoProgressCard";

export interface VideoTransectTabProps {
  uploadedVideo: UploadedVideoItem | null;
  onSelectVideoFile: (file: File) => void;
  onLoadSampleVideo: () => void;
  onRemoveVideo: () => void;
  videoError: string | null;
  isUploadingVideo: boolean;
  isStartingVideoJob: boolean;
  videoJobStatus: VideoJobStatus | null;
  isBackendConnected: boolean;
  isPausingResumingVideo: boolean;
  onStartVideoAnalysis: () => void;
  onPauseResumeVideo: () => void;
  onCancelVideo: () => void;
  onOpenVideoStudioFromPanel: () => void;
}

export const VideoTransectTab: React.FC<VideoTransectTabProps> = ({
  uploadedVideo,
  onSelectVideoFile,
  onLoadSampleVideo,
  onRemoveVideo,
  videoError,
  isUploadingVideo,
  isStartingVideoJob,
  videoJobStatus,
  isBackendConnected,
  isPausingResumingVideo,
  onStartVideoAnalysis,
  onPauseResumeVideo,
  onCancelVideo,
  onOpenVideoStudioFromPanel,
}) => {
  const [videoDragActive, setVideoDragActive] = useState<boolean>(false);
  const videoInputRef = useRef<HTMLInputElement>(null);

  return (
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
          onClick={onLoadSampleVideo}
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
            if (file) onSelectVideoFile(file);
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
            if (file) onSelectVideoFile(file);
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
                    {uploadedVideo.metadata.duration_sec}s ({uploadedVideo.metadata.total_frames}{" "}
                    frames)
                  </div>
                </div>
              </div>

              <Button
                size="sm"
                variant="ghost"
                onClick={onRemoveVideo}
                className="h-7 w-7 p-0 text-slate-400 hover:text-red-600"
              >
                <X size={14} />
              </Button>
            </div>

            <p className="pt-1 border-t border-cyan-100 text-[11px] text-slate-500">
              Colonies are segmented on keyframes and carried between them by camera-motion tracking, then
              classified once each for genus and bleaching.
            </p>
          </div>
        )}
      </div>

      {/* Fixed Bottom Launch & Transect Control Bar */}
      <VideoProgressCard
        videoJobStatus={videoJobStatus}
        uploadedVideo={uploadedVideo}
        isStartingVideoJob={isStartingVideoJob}
        isUploadingVideo={isUploadingVideo}
        isBackendConnected={isBackendConnected}
        isPausingResumingVideo={isPausingResumingVideo}
        onStartVideoAnalysis={onStartVideoAnalysis}
        onPauseResumeVideo={onPauseResumeVideo}
        onCancelVideo={onCancelVideo}
        onOpenVideoStudioFromPanel={onOpenVideoStudioFromPanel}
      />
    </div>
  );
};
