import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  X,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Minimize2,
  Layers,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Edit3,
} from "lucide-react";
import { CoralSegment } from "../types";

interface ImageLightboxModalProps {
  isOpen: boolean;
  onClose: () => void;
  primarySrc: string;
  primaryTitle: string;
  primaryBadge?: string;
  secondarySrc?: string | null;
  secondaryTitle?: string;
  secondaryBadge?: string;
  initialMode?: "primary" | "secondary";
  segments?: CoralSegment[];
  imageResolution?: string;
  onOpenEditor?: () => void;
}

export const ImageLightboxModal: React.FC<ImageLightboxModalProps> = ({
  isOpen,
  onClose,
  primarySrc,
  primaryTitle,
  primaryBadge = "SOURCE",
  secondarySrc,
  secondaryTitle = "Overlay",
  secondaryBadge = "OVERLAY",
  initialMode = "secondary",
  segments = [],
  imageResolution = "0x0",
  onOpenEditor,
}) => {
  const [activeMode, setActiveMode] = useState<"primary" | "secondary">(
    secondarySrc && initialMode === "secondary" ? "secondary" : "primary"
  );
  const [scale, setScale] = useState<number>(1);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Performance refs to bypass React rendering during drag & wheel gestures
  const scaleRef = useRef<number>(1);
  const positionRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const isDraggingRef = useRef<boolean>(false);
  const dragStartRef = useRef<{ startX: number; startY: number; posX: number; posY: number }>({
    startX: 0,
    startY: 0,
    posX: 0,
    posY: 0,
  });

  const dragRafRef = useRef<number | null>(null);
  const wheelRafRef = useRef<number | null>(null);
  const hoverRafRef = useRef<number | null>(null);

  // Hovered coral detected by mouse position
  const [hoveredSegment, setHoveredSegment] = useState<CoralSegment | null>(null);
  const hoveredSegmentRef = useRef<CoralSegment | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number }>({ w: 1920, h: 1080 });

  const containerRef = useRef<HTMLDivElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  // Apply transforms with GPU acceleration and optional animation
  const applyTransform = useCallback(
    (newScale: number, newPos: { x: number; y: number }, animate: boolean = true) => {
      scaleRef.current = newScale;
      positionRef.current = newPos;
      setScale(newScale);
      setPosition(newPos);

      if (wrapperRef.current) {
        wrapperRef.current.style.transition = animate
          ? "transform 0.18s cubic-bezier(0.2, 0, 0, 1)"
          : "none";
        wrapperRef.current.style.transform = `translate3d(${newPos.x}px, ${newPos.y}px, 0) scale(${newScale})`;
      }
    },
    []
  );

  const handleResetZoom = useCallback(() => {
    applyTransform(1, { x: 0, y: 0 }, true);
  }, [applyTransform]);

  const handleZoomIn = useCallback(() => {
    const nextScale = Math.min(Number((scaleRef.current + 0.35).toFixed(2)), 5.0);
    applyTransform(nextScale, positionRef.current, true);
  }, [applyTransform]);

  const handleZoomOut = useCallback(() => {
    const nextScale = Math.max(Number((scaleRef.current - 0.35).toFixed(2)), 0.6);
    const nextPos = nextScale <= 1 ? { x: 0, y: 0 } : positionRef.current;
    applyTransform(nextScale, nextPos, true);
  }, [applyTransform]);

  // Reset modal state on open or image switch
  useEffect(() => {
    if (isOpen) {
      setActiveMode(secondarySrc && initialMode === "secondary" ? "secondary" : "primary");
      scaleRef.current = 1;
      positionRef.current = { x: 0, y: 0 };
      setScale(1);
      setPosition({ x: 0, y: 0 });
      hoveredSegmentRef.current = null;
      setHoveredSegment(null);

      if (wrapperRef.current) {
        wrapperRef.current.style.transition = "none";
        wrapperRef.current.style.transform = "translate3d(0, 0, 0) scale(1)";
      }

      if (imageResolution && imageResolution.includes("x")) {
        const [rw, rh] = imageResolution.split("x").map(Number);
        if (rw > 0 && rh > 0) {
          setImgDims({ w: rw, h: rh });
        }
      }
    }
  }, [isOpen, initialMode, secondarySrc, imageResolution]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        handleZoomIn();
      } else if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        handleZoomOut();
      } else if (e.key === "0" || e.key.toLowerCase() === "r") {
        e.preventDefault();
        handleResetZoom();
      } else if (e.key === "Tab" && secondarySrc) {
        e.preventDefault();
        setActiveMode((prev) => (prev === "primary" ? "secondary" : "primary"));
        hoveredSegmentRef.current = null;
        setHoveredSegment(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose, handleZoomIn, handleZoomOut, handleResetZoom, secondarySrc]);

  // Global mouse handlers for butter-smooth panning (zero React re-render lag)
  useEffect(() => {
    if (!isOpen) return;

    const handleWindowMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;

      const dx = e.clientX - dragStartRef.current.startX;
      const dy = e.clientY - dragStartRef.current.startY;
      const nextX = dragStartRef.current.posX + dx;
      const nextY = dragStartRef.current.posY + dy;

      positionRef.current = { x: nextX, y: nextY };

      if (dragRafRef.current === null) {
        dragRafRef.current = requestAnimationFrame(() => {
          if (wrapperRef.current) {
            wrapperRef.current.style.transform = `translate3d(${positionRef.current.x}px, ${positionRef.current.y}px, 0) scale(${scaleRef.current})`;
            wrapperRef.current.style.transition = "none";
          }
          dragRafRef.current = null;
        });
      }
    };

    const handleWindowMouseUp = () => {
      if (isDraggingRef.current) {
        isDraggingRef.current = false;
        setIsDragging(false);

        if (dragRafRef.current !== null) {
          cancelAnimationFrame(dragRafRef.current);
          dragRafRef.current = null;
        }

        // Synchronize React state on drag completion
        setPosition({ ...positionRef.current });
      }
    };

    window.addEventListener("mousemove", handleWindowMouseMove, { passive: true });
    window.addEventListener("mouseup", handleWindowMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", handleWindowMouseUp);
      if (dragRafRef.current !== null) {
        cancelAnimationFrame(dragRafRef.current);
        dragRafRef.current = null;
      }
    };
  }, [isOpen]);

  // Smooth wheel & trackpad pinch zoom with cursor anchoring
  useEffect(() => {
    const stage = containerRef.current;
    if (!stage || !isOpen) return;

    const handleWheelNative = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();

      // Proportional factor: smooth on trackpads, responsive on scroll wheels
      let factor = 1;
      if (e.ctrlKey) {
        // macOS pinch gesture
        factor = Math.exp(-e.deltaY * 0.015);
      } else if (Math.abs(e.deltaY) < 40) {
        // macOS trackpad smooth scroll
        factor = Math.exp(-e.deltaY * 0.0035);
      } else {
        // Discrete mouse wheel notch
        factor = e.deltaY < 0 ? 1.22 : 0.82;
      }

      const currentScale = scaleRef.current;
      const nextScale = Math.min(Math.max(Number((currentScale * factor).toFixed(3)), 0.5), 6.0);

      if (Math.abs(nextScale - currentScale) < 0.001) return;

      // Anchor zoom relative to cursor position inside the stage
      const rect = stage.getBoundingClientRect();
      const cursorX = e.clientX - (rect.left + rect.width / 2);
      const cursorY = e.clientY - (rect.top + rect.height / 2);

      const ratio = nextScale / currentScale;
      let nextX = cursorX - (cursorX - positionRef.current.x) * ratio;
      let nextY = cursorY - (cursorY - positionRef.current.y) * ratio;

      if (nextScale <= 1) {
        nextX = 0;
        nextY = 0;
      }

      scaleRef.current = nextScale;
      positionRef.current = { x: nextX, y: nextY };

      // Direct GPU transform update for instant 60/120fps feedback
      if (wrapperRef.current) {
        wrapperRef.current.style.transform = `translate3d(${nextX}px, ${nextY}px, 0) scale(${nextScale})`;
        wrapperRef.current.style.transition = "none";
      }

      // Batch React UI state synchronization
      if (wheelRafRef.current !== null) {
        cancelAnimationFrame(wheelRafRef.current);
      }
      wheelRafRef.current = requestAnimationFrame(() => {
        setScale(nextScale);
        setPosition({ x: nextX, y: nextY });
        wheelRafRef.current = null;
      });
    };

    stage.addEventListener("wheel", handleWheelNative, { passive: false });
    return () => {
      stage.removeEventListener("wheel", handleWheelNative);
      if (wheelRafRef.current !== null) {
        cancelAnimationFrame(wheelRafRef.current);
        wheelRafRef.current = null;
      }
    };
  }, [isOpen]);

  // Stage Mouse Down to begin panning
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0 && e.button !== 1) return;
    e.preventDefault();

    isDraggingRef.current = true;
    setIsDragging(true);

    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      posX: positionRef.current.x,
      posY: positionRef.current.y,
    };

    // Hide coral telemetry while panning
    if (hoveredSegmentRef.current) {
      hoveredSegmentRef.current = null;
      setHoveredSegment(null);
    }
  };

  // Throttled coral detection under cursor (Zero layout thrashing)
  const handleStageMouseMove = (e: React.MouseEvent) => {
    if (isDraggingRef.current) return;
    if (activeMode !== "secondary" || !imgRef.current || segments.length === 0) {
      if (hoveredSegmentRef.current !== null) {
        hoveredSegmentRef.current = null;
        setHoveredSegment(null);
      }
      return;
    }

    const mouseX = e.clientX;
    const mouseY = e.clientY;

    if (hoverRafRef.current === null) {
      hoverRafRef.current = requestAnimationFrame(() => {
        hoverRafRef.current = null;
        if (isDraggingRef.current || !imgRef.current) return;

        const rect = imgRef.current.getBoundingClientRect();
        const relX = mouseX - rect.left;
        const relY = mouseY - rect.top;

        if (
          relX >= 0 &&
          relX <= rect.width &&
          relY >= 0 &&
          relY <= rect.height &&
          rect.width > 0 &&
          rect.height > 0
        ) {
          const scaleX = imgDims.w / rect.width;
          const scaleY = imgDims.h / rect.height;
          const origX = relX * scaleX;
          const origY = relY * scaleY;

          let matchedSeg: CoralSegment | null = null;
          let minDistance = 50;

          for (let i = 0; i < segments.length; i++) {
            const seg = segments[i];
            if (seg.bbox) {
              const [bx, by, bw, bh] = seg.bbox;
              if (origX >= bx && origX <= bx + bw && origY >= by && origY <= by + bh) {
                matchedSeg = seg;
                break;
              }
            } else if (seg.centroid) {
              const [cx, cy] = seg.centroid;
              const dist = Math.hypot(origX - cx, origY - cy);
              if (dist < minDistance) {
                minDistance = dist;
                matchedSeg = seg;
              }
            }
          }

          if (hoveredSegmentRef.current?.id !== matchedSeg?.id) {
            hoveredSegmentRef.current = matchedSeg;
            setHoveredSegment(matchedSeg);
          }
        } else {
          if (hoveredSegmentRef.current !== null) {
            hoveredSegmentRef.current = null;
            setHoveredSegment(null);
          }
        }
      });
    }
  };

  const handleStageMouseLeave = () => {
    if (hoveredSegmentRef.current !== null) {
      hoveredSegmentRef.current = null;
      setHoveredSegment(null);
    }
    if (hoverRafRef.current !== null) {
      cancelAnimationFrame(hoverRafRef.current);
      hoverRafRef.current = null;
    }
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (scaleRef.current > 1.2) {
      handleResetZoom();
    } else {
      applyTransform(2.5, positionRef.current, true);
    }
  };

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const naturalW = e.currentTarget.naturalWidth;
    const naturalH = e.currentTarget.naturalHeight;
    if (naturalW > 0 && naturalH > 0) {
      setImgDims({ w: naturalW, h: naturalH });
    }
  };

  if (!isOpen) return null;

  const currentDisplaySrc =
    activeMode === "secondary" && secondarySrc ? secondarySrc : primarySrc;
  const currentTitle =
    activeMode === "secondary" ? secondaryTitle : primaryTitle;
  const currentBadge =
    activeMode === "secondary" ? secondaryBadge : primaryBadge;

  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox-modal" onClick={(e) => e.stopPropagation()}>
        {/* Clean Theme Header */}
        <header className="lightbox-header">
          <div className="lightbox-header-left">
            <span className="lightbox-badge">{currentBadge}</span>
            <span className="lightbox-title">{currentTitle}</span>
            <span className="lightbox-zoom-chip">
              {Math.round(scale * 100)}%
            </span>
          </div>

          {/* Clean Segmented Control: Original vs Overlay */}
          {secondarySrc && (
            <div className="lightbox-mode-switch">
              <button
                type="button"
                className={`lightbox-switch-btn ${activeMode === "primary" ? "active" : ""}`}
                onClick={() => {
                  setActiveMode("primary");
                  hoveredSegmentRef.current = null;
                  setHoveredSegment(null);
                }}
              >
                <Layers size={13} />
                <span>Original</span>
              </button>
              <button
                type="button"
                className={`lightbox-switch-btn ${activeMode === "secondary" ? "active" : ""}`}
                onClick={() => setActiveMode("secondary")}
              >
                <Sparkles size={13} />
                <span>Overlay</span>
              </button>
            </div>
          )}

          {/* Header Right Actions */}
          <div className="lightbox-header-right flex items-center gap-2">
            {onOpenEditor && (
              <button
                type="button"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-600 hover:bg-teal-500 text-white shadow-xs cursor-pointer transition-colors"
                onClick={() => {
                  onClose();
                  onOpenEditor();
                }}
                title="Open Manual Segmentation & Mask Editor"
              >
                <Edit3 size={13} />
                <span>Edit Masks</span>
              </button>
            )}
            <button
              type="button"
              className="lightbox-close-btn"
              onClick={onClose}
              title="Close Fullscreen (Esc)"
            >
              <X size={18} />
            </button>
          </div>
        </header>

        {/* Clean Stage with Zero-Lag Dragging and Anchored Zoom */}
        <div
          ref={containerRef}
          className={`lightbox-stage ${isDragging ? "grabbing" : scale > 1 ? "grabbable" : ""}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleStageMouseMove}
          onMouseLeave={handleStageMouseLeave}
          onDoubleClick={handleDoubleClick}
        >
          <div
            ref={wrapperRef}
            className="lightbox-image-wrapper"
            style={{
              transform: `translate3d(${position.x}px, ${position.y}px, 0) scale(${scale})`,
              transformOrigin: "center center",
              transition: isDragging ? "none" : "transform 0.18s cubic-bezier(0.2, 0, 0, 1)",
            }}
          >
            <img
              ref={imgRef}
              src={currentDisplaySrc}
              alt={currentTitle}
              className="lightbox-img"
              draggable={false}
              onLoad={handleImageLoad}
            />
          </div>

          {/* Floating Coral Telemetry HUD Card (Appears cleanly when hovering a coral) */}
          {activeMode === "secondary" && hoveredSegment && (
            <div className="coral-hud-card" onClick={(e) => e.stopPropagation()}>
              <div className="coral-hud-header">
                <div
                  className="coral-hud-id"
                  style={{ backgroundColor: hoveredSegment.color_hex || "var(--color-navy)" }}
                >
                  {hoveredSegment.id_str}
                </div>
                <div className="coral-hud-names">
                  <span className="coral-hud-genus">{hoveredSegment.genus || "Coral"}</span>
                  <span className="coral-hud-form">{hoveredSegment.growth_form || "Colony"}</span>
                </div>
              </div>

              <div className="coral-hud-body">
                <div className="coral-hud-row">
                  <span className="hud-label">Condition</span>
                  {hoveredSegment.condition === "Healthy" ? (
                    <span className="hud-pill healthy">
                      <CheckCircle2 size={11} />
                      <span>Healthy ({hoveredSegment.condition_conf}%)</span>
                    </span>
                  ) : (
                    <span className="hud-pill bleached">
                      <AlertTriangle size={11} />
                      <span>Bleached ({hoveredSegment.condition_conf}%)</span>
                    </span>
                  )}
                </div>

                <div className="coral-hud-row">
                  <span className="hud-label">Coverage</span>
                  <span className="hud-val">
                    {hoveredSegment.area_pct}% ({hoveredSegment.area_px.toLocaleString()} px)
                  </span>
                </div>

                <div className="coral-hud-row">
                  <span className="hud-label">SAM IoU</span>
                  <span className="hud-val">{hoveredSegment.predicted_iou}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Clean Theme Footer Controls */}
        <footer className="lightbox-footer">
          <div className="lightbox-controls-dock">
            <button
              type="button"
              className="lightbox-dock-btn"
              onClick={handleZoomOut}
              disabled={scale <= 0.6}
              title="Zoom Out (-)"
            >
              <ZoomOut size={15} />
            </button>

            <span className="lightbox-dock-scale">
              {Math.round(scale * 100)}%
            </span>

            <button
              type="button"
              className="lightbox-dock-btn"
              onClick={handleZoomIn}
              disabled={scale >= 5.0}
              title="Zoom In (+)"
            >
              <ZoomIn size={15} />
            </button>

            <div className="lightbox-dock-separator" />

            <button
              type="button"
              className="lightbox-dock-btn text-btn"
              onClick={handleResetZoom}
              title="Reset Zoom to Fit (R or 0)"
            >
              <RotateCcw size={13} />
              <span>Reset</span>
            </button>

            {scale === 1 ? (
              <button
                type="button"
                className="lightbox-dock-btn text-btn"
                onClick={() => applyTransform(2.5, positionRef.current, true)}
                title="Zoom to 2.5x Details"
              >
                <Maximize2 size={13} />
                <span>2.5x</span>
              </button>
            ) : (
              <button
                type="button"
                className="lightbox-dock-btn text-btn"
                onClick={handleResetZoom}
                title="Fit to Screen"
              >
                <Minimize2 size={13} />
                <span>Fit</span>
              </button>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
};
