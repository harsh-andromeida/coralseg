import React, { useRef, useEffect } from "react";
import { Film, Sparkles } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ThreeDotsLoader } from "../ThreeDotsLoader";
import { CoralSegment } from "../../types";

interface VideoViewerProps {
  videoSrc: string;
  layoutMode: string;
  colorMode: string;
  alpha: number;
  drawContours: boolean;
  drawLabels: boolean;
  drawBoxes: boolean;
  segments: CoralSegment[];
  selectedSegmentId: number | null;
  videoResolution: string;
  isLoading?: boolean;
  currentTimeSec: number;
  isPlaying: boolean;
  onTimeUpdate: (curTime: number) => void;
  onVideoEnded?: () => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export const VideoViewer: React.FC<VideoViewerProps> = ({
  videoSrc,
  layoutMode,
  colorMode,
  alpha,
  drawContours,
  drawLabels,
  drawBoxes,
  segments,
  selectedSegmentId,
  videoResolution,
  isLoading = false,
  currentTimeSec,
  isPlaying,
  onTimeUpdate,
  onVideoEnded,
  videoRef,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const secondaryVideoRef = useRef<HTMLVideoElement>(null);

  // Sync secondary video for side-by-side playback
  useEffect(() => {
    if (layoutMode === "Side-by-Side" && secondaryVideoRef.current && videoRef.current) {
      if (Math.abs(secondaryVideoRef.current.currentTime - videoRef.current.currentTime) > 0.08) {
        secondaryVideoRef.current.currentTime = videoRef.current.currentTime;
      }
      if (isPlaying && secondaryVideoRef.current.paused) {
        secondaryVideoRef.current.play().catch(() => {});
      } else if (!isPlaying && !secondaryVideoRef.current.paused) {
        secondaryVideoRef.current.pause();
      }
    }
  }, [isPlaying, currentTimeSec, layoutMode]);

  // Render canvas overlays synchronized with video frame
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const vidW = video.videoWidth || 972;
    const vidH = video.videoHeight || 720;
    const scaleX = canvas.width / vidW;
    const scaleY = canvas.height / vidH;

    // 1. Draw Segmentation Polygons
    segments.forEach((seg) => {
      if (!seg.polygon || seg.polygon.length < 3) return;
      const isSelected = selectedSegmentId === seg.id;
      const hex = seg.color_hex || "#06b6d4";

      ctx.beginPath();
      ctx.moveTo(seg.polygon[0][0] * scaleX, seg.polygon[0][1] * scaleY);
      for (let i = 1; i < seg.polygon.length; i++) {
        ctx.lineTo(seg.polygon[i][0] * scaleX, seg.polygon[i][1] * scaleY);
      }
      ctx.closePath();

      // Semi-transparent fill
      const fillOpacity = Math.round(alpha * 255).toString(16).padStart(2, "0");
      ctx.fillStyle = `${hex}${fillOpacity}`;
      ctx.fill();

      // Outline
      if (drawContours) {
        ctx.strokeStyle = isSelected ? "#ffffff" : hex;
        ctx.lineWidth = isSelected ? 3.5 : 2;
        ctx.stroke();
      }
    });

    // 2. Draw Bounding Boxes
    if (drawBoxes) {
      segments.forEach((seg) => {
        if (!seg.bbox) return;
        const [x, y, bw, bh] = seg.bbox;
        const isSelected = selectedSegmentId === seg.id;
        const hex = seg.color_hex || "#06b6d4";

        ctx.strokeStyle = isSelected ? "#ffffff" : hex;
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.strokeRect(x * scaleX, y * scaleY, bw * scaleX, bh * scaleY);
      });
    }

    // 3. Draw Labels & Badges
    if (drawLabels) {
      ctx.font = "bold 11px system-ui, -apple-system, sans-serif";
      segments.forEach((seg) => {
        if (!seg.bbox) return;
        const [x, y] = seg.bbox;
        const label = `${seg.id_str || `#${seg.id}`} ${seg.genus} [${seg.condition}]`;
        const renderX = Math.max(4, x * scaleX);
        const renderY = Math.max(16, y * scaleY - 6);
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(renderX - 2, renderY - 12, textWidth + 8, 16);

        ctx.strokeStyle = seg.condition === "Bleached" ? "#ef4444" : seg.color_hex || "#06b6d4";
        ctx.lineWidth = 1;
        ctx.strokeRect(renderX - 2, renderY - 12, textWidth + 8, 16);

        ctx.fillStyle = seg.condition === "Bleached" ? "#f87171" : "#f1f5f9";
        ctx.fillText(label, renderX + 2, renderY);
      });
    }
  }, [segments, selectedSegmentId, alpha, drawContours, drawBoxes, drawLabels, currentTimeSec, layoutMode]);

  const isSideBySide = layoutMode === "Side-by-Side";
  const showOriginal = isSideBySide || layoutMode === "Original Only";
  const showOverlay = isSideBySide || layoutMode === "Overlay Only" || layoutMode === "Masks on Black";

  const getBadgeLabel = () => {
    if (colorMode === "taxonomy") return "TAXONOMY";
    if (colorMode === "bleaching") return "CONDITION";
    return "INSTANCES";
  };

  return (
    <div className={`grid gap-6 ${isSideBySide ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}>
      {/* Frame 1: Original Raw Video */}
      {showOriginal && (
        <Card className="overflow-hidden border-border/80 bg-card shadow-sm rounded-xl flex flex-col group transition-all duration-200 hover:shadow-md">
          <CardHeader className="py-3 px-5 flex flex-row items-center justify-between border-b border-border/70 bg-card">
            <CardTitle className="text-xs font-semibold uppercase tracking-wider text-foreground flex items-center gap-2">
              <Film className="h-4 w-4 text-primary" />
              <span>Original Transect Video</span>
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-[10px] font-mono bg-muted/50 border-border/60">
                RAW {videoResolution}
              </Badge>
            </div>
          </CardHeader>
          <div className="relative bg-muted/40 aspect-video flex items-center justify-center overflow-hidden">
            <video
              ref={isSideBySide ? secondaryVideoRef : (videoRef as any)}
              src={videoSrc}
              muted
              playsInline
              onTimeUpdate={(e) => {
                if (!isSideBySide) onTimeUpdate((e.target as HTMLVideoElement).currentTime);
              }}
              onEnded={onVideoEnded}
              className="max-h-[64vh] w-auto mx-auto object-contain select-none"
            />
          </div>
        </Card>
      )}

      {/* Frame 2: Annotated Overlay Video */}
      {showOverlay && (
        <Card className="overflow-hidden border-border/80 bg-card shadow-sm rounded-xl flex flex-col group transition-all duration-200 hover:shadow-md">
          <CardHeader className="py-3 px-5 flex flex-row items-center justify-between border-b border-border/70 bg-card">
            <CardTitle className="text-xs font-semibold uppercase tracking-wider text-foreground flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <span>Benthic Tracking Overlay</span>
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-[10px] font-mono">
                {getBadgeLabel()}
              </Badge>
              <Badge variant="outline" className="text-[10px] font-mono bg-muted/50 border-border/60">
                {segments.length} CORALS
              </Badge>
            </div>
          </CardHeader>
          <div className="relative bg-muted/40 aspect-video flex items-center justify-center overflow-hidden">
            <video
              ref={isSideBySide ? (videoRef as any) : undefined}
              src={videoSrc}
              muted
              playsInline
              onTimeUpdate={(e) => onTimeUpdate((e.target as HTMLVideoElement).currentTime)}
              onEnded={onVideoEnded}
              className="max-h-[64vh] w-auto mx-auto object-contain select-none"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none select-none"
            />
            {isLoading && (
              <div className="absolute inset-0 bg-background/50 backdrop-blur-xs flex flex-col items-center justify-center gap-2">
                <ThreeDotsLoader size="md" />
                <span className="text-xs font-mono text-muted-foreground">Processing Transect Track...</span>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
};
