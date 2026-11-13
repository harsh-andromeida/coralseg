import React, { useRef, useEffect, useState, useCallback } from "react";
import { Film, Sparkles, AlertCircle } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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

interface VideoRenderBox {
  left: number;
  top: number;
  width: number;
  height: number;
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
  const overlayContainerRef = useRef<HTMLDivElement>(null);
  const [renderBox, setRenderBox] = useState<VideoRenderBox>({ left: 0, top: 0, width: 0, height: 0 });
  const [videoError, setVideoError] = useState<string | null>(null);

  useEffect(() => {
    setVideoError(null);
  }, [videoSrc]);

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

  // Compute exact visible video bounding box inside container
  const updateRenderBox = useCallback(() => {
    const container = overlayContainerRef.current;
    const video = videoRef.current;
    if (!container || !video) return;

    const cRect = container.getBoundingClientRect();
    const vRect = video.getBoundingClientRect();

    const cW = video.clientWidth;
    const cH = video.clientHeight;
    const vW = video.videoWidth || cW;
    const vH = video.videoHeight || cH;

    if (cW <= 0 || cH <= 0 || vW <= 0 || vH <= 0) {
      setRenderBox({ left: 0, top: 0, width: Math.round(cRect.width), height: Math.round(cRect.height) });
      return;
    }

    const videoAspect = vW / vH;
    const elemAspect = cW / cH;

    let renderW = cW;
    let renderH = cH;
    let renderX = 0;
    let renderY = 0;

    // Detect if the video element has letterboxing/pillarboxing inside its CSS box
    if (elemAspect > videoAspect + 0.005) {
      renderW = cH * videoAspect;
      renderH = cH;
      renderX = (cW - renderW) / 2;
    } else if (elemAspect < videoAspect - 0.005) {
      renderH = cW / videoAspect;
      renderW = cW;
      renderY = (cH - renderH) / 2;
    }

    // Offset relative to the container element
    const left = vRect.left - cRect.left + renderX;
    const top = vRect.top - cRect.top + renderY;

    setRenderBox({
      left: Math.round(left),
      top: Math.round(top),
      width: Math.max(1, Math.round(renderW)),
      height: Math.max(1, Math.round(renderH)),
    });
  }, [videoRef]);

  // Observe container & video resizing to maintain pixel-perfect overlay registration
  useEffect(() => {
    updateRenderBox();
    const container = overlayContainerRef.current;
    const video = videoRef.current;
    if (!container || !video) return;

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => {
        updateRenderBox();
      });
      ro.observe(container);
      ro.observe(video);
    }

    window.addEventListener("resize", updateRenderBox);
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", updateRenderBox);
    };
  }, [updateRenderBox, videoRef, layoutMode]);

  // Render canvas overlays synchronized with video frame
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || renderBox.width <= 0 || renderBox.height <= 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const targetW = Math.round(renderBox.width * dpr);
    const targetH = Math.round(renderBox.height * dpr);

    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, renderBox.width, renderBox.height);

    if (layoutMode === "Masks on Black") {
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, renderBox.width, renderBox.height);
    }

    const vidW = video.videoWidth || 972;
    const vidH = video.videoHeight || 720;
    const scaleX = renderBox.width / vidW;
    const scaleY = renderBox.height / vidH;

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
      const placed: { x: number; y: number; w: number; h: number }[] = [];
      const labelH = 16;
      segments.forEach((seg) => {
        if (!seg.bbox) return;
        const [x, y] = seg.bbox;
        const label = `${seg.id_str || `#${seg.id}`} ${seg.genus} [${seg.condition}]`;
        const textWidth = ctx.measureText(label).width;
        const boxW = textWidth + 8;
        const renderX = Math.min(Math.max(4, x * scaleX), Math.max(4, renderBox.width - boxW - 2));
        // Anchor above the coral, then nudge down past any label already placed so none stack on each other
        let boxY = Math.max(2, y * scaleY - 6 - 12);
        for (let guard = 0; guard < 40; guard++) {
          const hit = placed.find(
            (p) => renderX - 2 < p.x + p.w && renderX - 2 + boxW > p.x && boxY < p.y + p.h && boxY + labelH > p.y,
          );
          if (!hit) break;
          boxY = hit.y + hit.h + 1;
        }
        boxY = Math.min(boxY, Math.max(2, renderBox.height - labelH - 2));
        placed.push({ x: renderX - 2, y: boxY, w: boxW, h: labelH });

        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(renderX - 2, boxY, boxW, labelH);

        ctx.strokeStyle = seg.condition === "Bleached" ? "#ef4444" : seg.color_hex || "#06b6d4";
        ctx.lineWidth = 1;
        ctx.strokeRect(renderX - 2, boxY, boxW, labelH);

        ctx.fillStyle = seg.condition === "Bleached" ? "#f87171" : "#f1f5f9";
        ctx.fillText(label, renderX + 2, boxY + 12);
      });
    }

    ctx.restore();
  }, [segments, selectedSegmentId, alpha, drawContours, drawBoxes, drawLabels, currentTimeSec, layoutMode, renderBox]);

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
              ref={isSideBySide ? secondaryVideoRef : (!showOverlay ? (videoRef as any) : undefined)}
              src={videoSrc}
              muted
              playsInline
              preload="auto"
              crossOrigin="anonymous"
              onLoadedData={() => {
                setVideoError(null);
                updateRenderBox();
              }}
              onError={(e) => {
                const mediaErr = (e.target as HTMLVideoElement).error;
                const detail = mediaErr ? `(Code ${mediaErr.code}: ${mediaErr.message || "Codec unsupported or stream unreachable"})` : "";
                setVideoError(`Video playback error ${detail}`);
              }}
              onTimeUpdate={(e) => {
                if (!isSideBySide) onTimeUpdate((e.target as HTMLVideoElement).currentTime);
              }}
              onEnded={onVideoEnded}
              className="max-h-[64vh] w-auto mx-auto object-contain select-none"
            />
            {videoError && (
              <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center text-white z-20">
                <AlertCircle className="w-8 h-8 text-rose-400 mb-2" />
                <p className="font-semibold text-sm">Failed to Load Video Stream</p>
                <p className="text-xs text-slate-300 font-mono mt-1 max-w-md">{videoError}</p>
                <p className="text-[11px] text-slate-400 mt-2">Source: <span className="font-mono text-cyan-300">{videoSrc}</span></p>
              </div>
            )}
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
          <div
            ref={overlayContainerRef}
            className="relative bg-muted/40 aspect-video flex items-center justify-center overflow-hidden"
          >
            <video
              ref={isSideBySide ? (videoRef as any) : (showOverlay ? (videoRef as any) : undefined)}
              src={videoSrc}
              muted
              playsInline
              preload="auto"
              crossOrigin="anonymous"
              onLoadedMetadata={updateRenderBox}
              onLoadedData={() => {
                setVideoError(null);
                updateRenderBox();
              }}
              onError={(e) => {
                const mediaErr = (e.target as HTMLVideoElement).error;
                const detail = mediaErr ? `(Code ${mediaErr.code}: ${mediaErr.message || "Codec unsupported or stream unreachable"})` : "";
                setVideoError(`Video playback error ${detail}`);
              }}
              onTimeUpdate={(e) => onTimeUpdate((e.target as HTMLVideoElement).currentTime)}
              onEnded={onVideoEnded}
              className="max-h-[64vh] w-auto mx-auto object-contain select-none"
            />
            {videoError && (
              <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center text-white z-20">
                <AlertCircle className="w-8 h-8 text-rose-400 mb-2" />
                <p className="font-semibold text-sm">Failed to Load Video Stream</p>
                <p className="text-xs text-slate-300 font-mono mt-1 max-w-md">{videoError}</p>
                <p className="text-[11px] text-slate-400 mt-2">Source: <span className="font-mono text-cyan-300">{videoSrc}</span></p>
              </div>
            )}
            {renderBox.width > 0 && renderBox.height > 0 && (
              <canvas
                ref={canvasRef}
                style={{
                  position: "absolute",
                  left: `${renderBox.left}px`,
                  top: `${renderBox.top}px`,
                  width: `${renderBox.width}px`,
                  height: `${renderBox.height}px`,
                }}
                className="pointer-events-none select-none"
              />
            )}
            {isLoading && (
              <div className="absolute top-3 right-3 bg-card/85 backdrop-blur-md border border-emerald-500/50 px-3 py-1 rounded-full flex items-center gap-2 shadow-sm pointer-events-none z-10">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-[11px] font-mono font-medium text-foreground tracking-wide">LIVE INFERENCE</span>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
};
