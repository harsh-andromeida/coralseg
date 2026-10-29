import React from "react";
import {
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Film,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { truncateMiddle } from "../PaginationBar";

interface VideoTimelineBarProps {
  currentFrame: number;
  totalFrames: number;
  currentTimeSec: number;
  durationSec: number;
  isPlaying: boolean;
  videoName: string;
  keyframeIndices?: number[];
  onTogglePlay: () => void;
  onStepFrame: (dir: -1 | 1) => void;
  onSeekFrame: (frame: number) => void;
}

export const VideoTimelineBar: React.FC<VideoTimelineBarProps> = ({
  currentFrame,
  totalFrames,
  currentTimeSec,
  durationSec,
  isPlaying,
  videoName,
  keyframeIndices = [],
  onTogglePlay,
  onStepFrame,
  onSeekFrame,
}) => {
  const formatTime = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms}`;
  };

  const isCurrentKeyframe = keyframeIndices.includes(currentFrame);

  return (
    <nav
      className="flex flex-col sm:flex-row items-center justify-between w-full min-h-[52px] px-4 py-2.5 rounded-xl border border-border/80 bg-card shadow-xs select-none gap-3"
      aria-label="Video Transect Scrubber"
    >
      {/* Left: Video file badge + frame index + 25 FPS badge */}
      <div className="flex items-center gap-2.5 min-w-0">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/60 border border-border/60 shadow-2xs min-w-0">
          <Film className="size-3.5 text-primary shrink-0" />
          <span
            className="font-mono text-xs font-semibold text-foreground truncate max-w-[140px] sm:max-w-[200px]"
            title={videoName}
          >
            {truncateMiddle(videoName, 24)}
          </span>
          <span className="text-muted-foreground/40 shrink-0">•</span>
          <span className="text-xs text-muted-foreground font-medium shrink-0 whitespace-nowrap">
            <strong className="text-foreground font-bold">{currentFrame + 1}</strong> of {totalFrames}
          </span>
          <span className="text-muted-foreground/40 shrink-0">•</span>
          <span className="font-mono text-xs text-primary font-semibold shrink-0">
            {formatTime(currentTimeSec)} / {formatTime(durationSec)}
          </span>
        </div>

        <Badge
          variant="outline"
          className="text-[10px] font-mono px-2 py-0.5 bg-cyan-500/10 text-cyan-500 dark:text-cyan-400 border-cyan-500/30 shrink-0"
        >
          25 FPS CFR
        </Badge>

        {isCurrentKeyframe && (
          <Badge
            variant="outline"
            className="text-[10px] font-mono px-2 py-0.5 bg-amber-500/10 text-amber-500 dark:text-amber-400 border-amber-500/30 shrink-0 flex items-center gap-1"
          >
            <Sparkles className="size-2.5" /> Keyframe
          </Badge>
        )}
      </div>

      {/* Center & Right: Transport controls + interactive scrubber slider */}
      <div className="flex items-center gap-2.5 w-full sm:w-auto flex-1 max-w-xl justify-end">
        {/* Step Backward */}
        <Button
          variant="outline"
          size="icon"
          onClick={() => onStepFrame(-1)}
          disabled={currentFrame <= 0}
          className="h-8 w-8 shrink-0 border-border/80"
          title="Previous Frame (Left Arrow)"
        >
          <ChevronLeft className="size-4" />
        </Button>

        {/* Play / Pause */}
        <Button
          variant="default"
          size="sm"
          onClick={onTogglePlay}
          className="h-8 px-3 gap-1.5 font-semibold shrink-0"
          title={isPlaying ? "Pause Video (Space)" : "Play Video (Space)"}
        >
          {isPlaying ? (
            <>
              <Pause className="size-3.5 fill-current" />
              <span>Pause</span>
            </>
          ) : (
            <>
              <Play className="size-3.5 fill-current" />
              <span>Play</span>
            </>
          )}
        </Button>

        {/* Step Forward */}
        <Button
          variant="outline"
          size="icon"
          onClick={() => onStepFrame(1)}
          disabled={currentFrame >= totalFrames - 1}
          className="h-8 w-8 shrink-0 border-border/80"
          title="Next Frame (Right Arrow)"
        >
          <ChevronRight className="size-4" />
        </Button>

        {/* Interactive Frame Slider */}
        <div className="flex-1 min-w-[120px] px-1 relative flex items-center">
          <Slider
            value={[currentFrame]}
            min={0}
            max={Math.max(1, totalFrames - 1)}
            step={1}
            onValueChange={(vals) => {
              if (vals.length > 0) onSeekFrame(vals[0]);
            }}
            className="cursor-pointer"
          />
        </div>
      </div>
    </nav>
  );
};
