import React, { useState } from "react";
import { Maximize2, Image as ImageIcon, Sparkles, Edit3 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ImageLightboxModal } from "./ImageLightboxModal";
import { ThreeDotsLoader } from "./ThreeDotsLoader";
import { CoralSegment, AnalysisStage } from "../types";
import { cn } from "@/lib/utils";

interface ImageViewerProps {
  originalSrc: string;
  overlaySrc: string | null;
  layoutMode: string;
  colorMode: string;
  isLoading?: boolean;
  analysisStage?: AnalysisStage;
  segments?: CoralSegment[];
  imageResolution?: string;
  onOpenEditor?: () => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  originalSrc,
  overlaySrc,
  layoutMode,
  colorMode,
  isLoading = false,
  analysisStage = "idle",
  segments = [],
  imageResolution = "0x0",
  onOpenEditor,
}) => {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxMode, setLightboxMode] = useState<"primary" | "secondary">("secondary");

  const openModal = (mode: "primary" | "secondary") => {
    setLightboxMode(mode);
    setLightboxOpen(true);
  };

  const getBadgeLabel = () => {
    if (colorMode === "taxonomy_condition") return "TAXONOMY + HEALTH";
    if (colorMode === "taxonomy") return "TAXONOMY";
    if (colorMode === "bleaching") return "CONDITION";
    return "INSTANCES";
  };

  const isSideBySide = layoutMode === "Side-by-Side";
  const showOriginal = isSideBySide || layoutMode === "Original Only";
  const showOverlay =
    isSideBySide || layoutMode === "Overlay Only" || layoutMode === "Masks on Black";
  const isDarkBg = layoutMode === "Masks on Black";

  const renderFrame = (
    title: string,
    badge: string,
    src: string | null,
    isDark: boolean,
    onOpen: () => void,
    isOverlay: boolean
  ) => (
    <Card className="overflow-hidden border-border/80 bg-card shadow-sm rounded-xl flex flex-col group transition-all duration-200 hover:shadow-md">
      <CardHeader className="py-3 px-5 flex flex-row items-center justify-between border-b border-border/70 bg-card">
        <CardTitle className="text-xs font-semibold uppercase tracking-wider text-foreground flex items-center gap-2">
          <ImageIcon className="h-4 w-4 text-primary" />
          <span>{title}</span>
        </CardTitle>
        <div className="flex items-center gap-2">
          {isOverlay && onOpenEditor && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onOpenEditor();
              }}
              className="h-6 px-2 text-[10px] font-semibold gap-1 bg-primary/10 text-primary border-primary/25 hover:bg-primary/20 cursor-pointer"
              title="Edit masks, adjust vertices, and annotate new corals"
            >
              <Edit3 className="h-3 w-3" />
              <span>Edit Masks</span>
            </Button>
          )}
          <Badge
            variant="outline"
            className={cn(
              "text-[10px] font-mono font-semibold uppercase px-2.5 py-0.5",
              isOverlay
                ? "bg-primary/10 text-primary border-primary/25"
                : "bg-muted/70 text-muted-foreground border-border/80"
            )}
          >
            {badge}
          </Badge>
        </div>
      </CardHeader>
      <CardContent
        className={cn(
          "p-0 relative flex items-center justify-center overflow-hidden select-none transition-colors",
          isDark
            ? "bg-slate-950"
            : "bg-slate-900/[0.03]"
        )}
      >
        {src ? (
          <>
            <img
              src={src}
              alt={title}
              className={cn(
                "w-full h-auto max-h-[68vh] object-contain cursor-pointer transition-opacity duration-300 block",
                isLoading && isOverlay ? "opacity-25 blur-[1px]" : "opacity-100"
              )}
              onClick={onOpen}
            />
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onOpen();
              }}
              className="absolute top-3 right-3 h-8 px-3 text-xs font-medium gap-1.5 opacity-0 group-hover:opacity-100 transition-all duration-200 bg-card/90 hover:bg-card text-foreground border border-border shadow-sm backdrop-blur-xs rounded-lg"
              title="Inspect Fullscreen"
            >
              <Maximize2 className="h-3.5 w-3.5 text-primary" />
              <span>Fullscreen</span>
            </Button>
          </>
        ) : (
          <div className="text-xs text-muted-foreground font-medium py-24">
            No Imagery Available
          </div>
        )}

        {isLoading && isOverlay && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 bg-card/85 backdrop-blur-xs text-center z-20 animate-fade-in">
            <ThreeDotsLoader size="lg" color="#0d7c85" />
            <div className="mt-4 inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary font-mono text-xs font-semibold shadow-xs">
              <Sparkles className="h-3.5 w-3.5" />
              <span>
                {analysisStage === "classifying"
                  ? "Step 2 of 3 • Taxonomy & Condition"
                  : analysisStage === "rendering"
                  ? "Step 3 of 3 • Synthesis & Rendering"
                  : "Step 1 of 3 • Colony Segmentation"}
              </span>
            </div>
            <div className="mt-2 text-sm font-semibold text-foreground">
              {analysisStage === "classifying"
                ? "Classifying Coral Taxa (BioCLIP & YOLO11)..."
                : analysisStage === "rendering"
                ? "Synthesizing Spatial Masks & Telemetry..."
                : "Segmenting colonies..."}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="w-full">
      <div
        className={cn(
          "grid gap-5",
          isSideBySide ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"
        )}
      >
        {showOriginal &&
          renderFrame(
            "Original Image",
            "SOURCE",
            originalSrc,
            false,
            () => openModal("primary"),
            false
          )}
        {showOverlay &&
          renderFrame(
            layoutMode === "Masks on Black"
              ? "Isolated Coral Masks"
              : "Segmentation Overlay",
            layoutMode === "Masks on Black" ? "MASKS" : getBadgeLabel(),
            overlaySrc || originalSrc,
            isDarkBg,
            () => openModal(overlaySrc ? "secondary" : "primary"),
            true
          )}
      </div>

      <ImageLightboxModal
        isOpen={lightboxOpen}
        onClose={() => setLightboxOpen(false)}
        primarySrc={originalSrc}
        primaryTitle="Original Reef Imagery"
        primaryBadge="SOURCE"
        secondarySrc={overlaySrc}
        secondaryTitle="Segmentation Overlay"
        secondaryBadge={getBadgeLabel()}
        initialMode={lightboxMode}
        segments={segments}
        imageResolution={imageResolution}
        onOpenEditor={onOpenEditor}
      />
    </div>
  );
};
