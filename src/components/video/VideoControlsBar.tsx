import React from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Sparkles,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface VideoControlsBarProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  onSeek: (timeSec: number) => void;
  onStepFrame: (direction: -1 | 1) => void;
  currentTime: number;
  duration: number;
  currentFrame: number;
  totalFrames: number;
  fps: number;
  realtimeFps: number;
  keyframeIndices: number[];
  playbackRate: number;
  onChangePlaybackRate: (rate: number) => void;
}

export const VideoControlsBar: React.FC<VideoControlsBarProps> = ({
  isPlaying,
  onTogglePlay,
  onSeek,
  onStepFrame,
  currentTime,
  duration,
  currentFrame,
  totalFrames,
  fps,
  realtimeFps,
  keyframeIndices,
  playbackRate,
  onChangePlaybackRate,
}) => {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 shadow-lg backdrop-blur-md flex flex-col gap-2.5">
      {/* Timeline Scrubber with Keyframe ticks */}
      <div className="relative w-full flex items-center group cursor-pointer">
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={currentTime}
          onChange={(e) => onSeek(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 hover:h-2.5 transition-all"
        />
        {/* Render keyframe markers along the scrubber */}
        {totalFrames > 0 &&
          keyframeIndices.map((kfIdx) => {
            const leftPct = (kfIdx / totalFrames) * 100;
            return (
              <div
                key={kfIdx}
                title={`Keyframe #${kfIdx}`}
                style={{ left: `${leftPct}%` }}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek(kfIdx / maxFps(fps));
                }}
                className="absolute top-0 w-1 h-2 bg-amber-400/80 rounded-full pointer-events-auto hover:scale-150 hover:bg-amber-300 transition-transform"
              />
            );
          })}
      </div>

      {/* Playback Controls & Status Badges */}
      <div className="flex items-center justify-between text-xs text-slate-300">
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onSeek(0)}
            title="Restart video"
            className="h-8 w-8 p-0 text-slate-400 hover:text-white"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => onStepFrame(-1)}
            title="Step backward 1 frame"
            className="h-8 w-8 p-0 text-slate-400 hover:text-white"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </Button>

          <Button
            size="sm"
            onClick={onTogglePlay}
            className={`h-8 px-3 gap-1.5 font-medium ${
              isPlaying
                ? "bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 border border-amber-500/30"
                : "bg-cyan-500 text-slate-950 hover:bg-cyan-400 font-semibold"
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="w-3.5 h-3.5 fill-current" /> Pause
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" /> Play Live
              </>
            )}
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => onStepFrame(1)}
            title="Step forward 1 frame"
            className="h-8 w-8 p-0 text-slate-400 hover:text-white"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </Button>

          <div className="font-mono text-xs text-slate-400 ml-2">
            <span className="text-white font-medium">{formatTime(currentTime)}</span> /{" "}
            <span>{formatTime(duration)}</span>
            <span className="text-slate-500 ml-1.5">
              (Frame {currentFrame}/{totalFrames})
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Keyframe Indicator */}
          {keyframeIndices.includes(currentFrame) && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 text-[11px] font-semibold animate-pulse">
              <Sparkles className="w-3 h-3 text-amber-400" />
              Keyframe Deep AI Pass
            </span>
          )}

          {/* Realtime Speed Pill */}
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700 font-mono text-[11px]">
            <Zap className="w-3 h-3 text-cyan-400" />
            <span className="text-slate-400">Processing:</span>
            <span className="text-cyan-300 font-bold">{realtimeFps.toFixed(1)} FPS</span>
          </div>

          {/* Speed Selector */}
          <select
            value={playbackRate}
            onChange={(e) => onChangePlaybackRate(parseFloat(e.target.value))}
            className="bg-slate-800 border border-slate-700 text-slate-300 rounded px-2 py-1 text-xs cursor-pointer hover:border-slate-600 focus:outline-none"
          >
            <option value={0.5}>0.5x</option>
            <option value={1.0}>1.0x</option>
            <option value={2.0}>2.0x</option>
            <option value={4.0}>4.0x</option>
          </select>
        </div>
      </div>
    </div>
  );
};

function maxFps(f: number) {
  return f > 0 ? f : 30.0;
}
