import React, { useState, useEffect, useRef, useCallback } from "react";
import Konva from "konva";
import { PromptPoint } from "../types";
import { promptSegment, warmupSamPrompt } from "../services/api";

import {
  TAXONOMY_OPTIONS,
  GROWTH_FORMS,
  DISTINCT_COLORS,
} from "./segmentation/constants";
import {
  ToolMode,
  EditableSegment,
  SegmentationEditorModalProps,
} from "./segmentation/types";
import {
  getUniquePolygonPoints,
  ensureClosedPolygon,
  findClosestEdge,
  calculatePolygonArea,
} from "./segmentation/geometry";
import { useEditorHistory } from "./segmentation/useEditorHistory";
import { useEditorHotkeys } from "./segmentation/useEditorHotkeys";
import { useSegmentManager } from "./segmentation/useSegmentManager";
import { EditorHeader } from "./segmentation/EditorHeader";
import { EditorCanvas } from "./segmentation/EditorCanvas";
import { EditorCanvasControls } from "./segmentation/EditorCanvasControls";
import { SamHudBanner } from "./segmentation/SamHudBanner";
import { PolygonHudBanner } from "./segmentation/PolygonHudBanner";
import { EditorSidebar } from "./segmentation/EditorSidebar";
import { SamLoadingOverlay } from "./segmentation/SamLoadingOverlay";
import { EditorShortcutsDialog } from "./segmentation/EditorShortcutsDialog";

// Re-export for external consumers
export {
  TAXONOMY_OPTIONS,
  GROWTH_FORMS,
  DISTINCT_COLORS,
  getUniquePolygonPoints,
  ensureClosedPolygon,
  findClosestEdge,
};
export type { ToolMode, EditableSegment, SegmentationEditorModalProps };

export const SegmentationEditorModal: React.FC<SegmentationEditorModalProps> = ({
  isOpen,
  onClose,
  imageName,
  imageSrc,
  imageResolution = "0x0",
  initialSegments = [],
  onSaveSuccess,
  onNavigatePrev,
  onNavigateNext,
  hasPrev = false,
  hasNext = false,
}) => {
  // Canvas image element
  const [htmlImage, setHtmlImage] = useState<HTMLImageElement | null>(null);
  const [imgDimensions, setImgDimensions] = useState<{ w: number; h: number }>({
    w: 1920,
    h: 1080,
  });

  // Active Tool Mode: 'select' (V), 'sam' (W), 'polygon' (P)
  const [toolMode, setToolMode] = useState<ToolMode>("select");

  // SAM Smart Click state
  const [samPoints, setSamPoints] = useState<PromptPoint[]>([]);
  const [samPointType, setSamPointType] = useState<1 | 0>(1); // 1 = Foreground (+), 0 = Background (-)
  const [isSamLoading, setIsSamLoading] = useState<boolean>(false);
  const [samPendingPolygon, setSamPendingPolygon] = useState<[number, number][] | null>(null);

  // Manual Polygon Draw state
  const [drawPoints, setDrawPoints] = useState<[number, number][]>([]);
  const [mouseCanvasPos, setMouseCanvasPos] = useState<[number, number] | null>(null);

  // Form inputs for active or new coral
  const [formGenus, setFormGenus] = useState<string>("Acropora");
  const [formGrowthForm, setFormGrowthForm] = useState<string>("Branching");
  const [formCondition, setFormCondition] = useState<"Healthy" | "Bleached">("Healthy");

  // Canvas Viewport transform (zoom / pan)
  const [stageScale, setStageScale] = useState<number>(1);
  const [stagePos, setStagePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [maskOpacity, setMaskOpacity] = useState<number>(0.45);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState<boolean>(false);

  const stageRef = useRef<Konva.Stage | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const loadedImageRef = useRef<string | null>(null);
  const wheelRafRef = useRef<number | null>(null);
  const samAbortControllerRef = useRef<AbortController | null>(null);
  const samTimeoutRef = useRef<number | null>(null);

  // History stack management
  const handleStateRestored = useCallback((restored: EditableSegment[]) => {
    setSegments(restored);
    segmentsRef.current = restored;
    setSelectedVertexIdx(null);
  }, []);

  const markUnsaved = useCallback(() => {
    hasUnsavedRef.current = true;
  }, []);

  const { resetHistory, pushHistory, undo, redo, canUndo, canRedo } = useEditorHistory({
    onStateRestored: handleStateRestored,
    markUnsaved,
  });

  // Segments and persistence manager
  const {
    segments,
    setSegments,
    segmentsRef,
    selectedId,
    setSelectedId,
    selectedVertexIdx,
    setSelectedVertexIdx,
    editorNotice,
    setEditorNotice,
    isSaving,
    saveStatus,
    hasUnsavedChanges,
    lastSavedTime,
    hasUnsavedRef,
    isSavingRef,
    loadPolygons,
    executeSave,
    addSegment,
    handleVertexDrag,
    handleVertexDragEnd,
    handleAddVertexOnEdge,
    handleDeleteVertex,
    handleDeleteSelected,
    handleDeleteCoral,
    handleUpdateSelectedMetadata,
    handleToggleVisibility,
  } = useSegmentManager({
    imageName,
    imageSrc,
    initialSegments,
    imgDimensions,
    onSaveSuccess,
    resetHistory,
    pushHistory,
  });

  // Undo wrapper (handling SAM uncommitted points)
  const handleUndo = useCallback(() => {
    if (toolMode === "sam" && samPoints.length > 0 && !samPendingPolygon) {
      setSamPoints((prev) => prev.slice(0, -1));
      return;
    }
    undo();
  }, [toolMode, samPoints.length, samPendingPolygon, undo]);

  // Load HTML image
  useEffect(() => {
    if (!isOpen || !imageSrc) return;
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = imageSrc;
    img.onload = () => {
      setHtmlImage(img);
      setImgDimensions({ w: img.naturalWidth, h: img.naturalHeight });
    };
  }, [isOpen, imageSrc]);

  // Periodic Auto-save every 20 seconds
  useEffect(() => {
    if (!isOpen) return;

    const interval = setInterval(() => {
      if (hasUnsavedRef.current && !isSavingRef.current) {
        executeSave(false);
      }
    }, 20000);

    return () => clearInterval(interval);
  }, [isOpen, executeSave, hasUnsavedRef, isSavingRef]);

  // Fit image to canvas container
  const handleFitToScreen = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth === 0 || clientHeight === 0) return;

    const scaleX = (clientWidth - 48) / imgDimensions.w;
    const scaleY = (clientHeight - 48) / imgDimensions.h;
    const fitScale = Math.min(scaleX, scaleY, 1.0);

    const posX = (clientWidth - imgDimensions.w * fitScale) / 2;
    const posY = (clientHeight - imgDimensions.h * fitScale) / 2;

    const finalScale = Number(fitScale.toFixed(3));
    const finalPos = { x: Math.round(posX), y: Math.round(posY) };

    const stage = stageRef.current;
    if (stage) {
      stage.scale({ x: finalScale, y: finalScale });
      stage.position(finalPos);
      stage.batchDraw();
    }

    setStageScale(finalScale);
    setStagePos(finalPos);
  }, [imgDimensions]);

  // Center-anchored zoom for toolbar +/- controls
  const handleZoomCenter = useCallback((factor: number) => {
    const stage = stageRef.current;
    const container = containerRef.current;
    if (!stage || !container) return;

    const oldScale = stage.scaleX();
    const newScale = Math.min(Math.max(Number((oldScale * factor).toFixed(3)), 0.2), 8.0);
    if (Math.abs(newScale - oldScale) < 0.001) return;

    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;

    const centerPointTo = {
      x: (centerX - stage.x()) / oldScale,
      y: (centerY - stage.y()) / oldScale,
    };

    const newPos = {
      x: Math.round(centerX - centerPointTo.x * newScale),
      y: Math.round(centerY - centerPointTo.y * newScale),
    };

    stage.scale({ x: newScale, y: newScale });
    stage.position(newPos);
    stage.batchDraw();

    setStageScale(newScale);
    setStagePos(newPos);
  }, []);

  // Auto-resize stage when container dimensions change
  useEffect(() => {
    if (!isOpen) return;
    const container = containerRef.current;
    if (!container) return;

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          const stage = stageRef.current;
          if (stage && (stage.width() !== width || stage.height() !== height)) {
            stage.width(width);
            stage.height(height);
            stage.batchDraw();
          }
        }
      }
    });

    ro.observe(container);
    return () => ro.disconnect();
  }, [isOpen]);

  // Cleanup pending wheel and SAM timers on unmount
  useEffect(() => {
    return () => {
      if (wheelRafRef.current !== null) {
        clearTimeout(wheelRafRef.current);
      }
      if (samTimeoutRef.current !== null) {
        window.clearTimeout(samTimeoutRef.current);
      }
      if (samAbortControllerRef.current) {
        samAbortControllerRef.current.abort("unmount");
      }
    };
  }, []);

  // Initialize and clean up modal image state
  useEffect(() => {
    if (isOpen) {
      if (loadedImageRef.current !== imageName) {
        loadedImageRef.current = imageName;
        loadPolygons();
        setToolMode("select");
        setSamPoints([]);
        setSamPendingPolygon(null);
        setDrawPoints([]);
        handleFitToScreen();
        warmupSamPrompt(imageName, imageSrc);
      }
    } else {
      if (samTimeoutRef.current !== null) {
        window.clearTimeout(samTimeoutRef.current);
        samTimeoutRef.current = null;
      }
      if (samAbortControllerRef.current) {
        samAbortControllerRef.current.abort("close");
        samAbortControllerRef.current = null;
      }
      setIsSamLoading(false);

      if (hasUnsavedRef.current && !isSavingRef.current) {
        executeSave(false);
      }
      loadedImageRef.current = null;
    }
  }, [isOpen, imageName, loadPolygons, handleFitToScreen, imageSrc, executeSave, hasUnsavedRef, isSavingRef]);

  // Synchronize form inputs when selected coral changes
  useEffect(() => {
    if (selectedId !== null) {
      const target = segments.find((s) => s.id === selectedId);
      if (target) {
        setFormGenus(target.genus);
        setFormGrowthForm(target.growth_form);
        setFormCondition(target.condition);
      }
    }
  }, [selectedId, segments]);

  // Mouse wheel zoom anchored to cursor with physics calibration & direct transform
  const handleWheel = useCallback((e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    // Proportional exponential factor: smooth on trackpads, responsive on scroll wheels
    let factor = 1;
    if (e.evt.ctrlKey) {
      // macOS pinch gesture
      factor = Math.exp(-e.evt.deltaY * 0.015);
    } else if (Math.abs(e.evt.deltaY) < 40) {
      // macOS trackpad smooth scroll
      factor = Math.exp(-e.evt.deltaY * 0.0035);
    } else {
      // Discrete mouse wheel notch
      factor = e.evt.deltaY < 0 ? 1.15 : 0.87;
    }

    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const newScale = Math.min(Math.max(Number((oldScale * factor).toFixed(3)), 0.2), 8.0);
    if (Math.abs(newScale - oldScale) < 0.001) return;

    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };

    const newPos = {
      x: Math.round(pointer.x - mousePointTo.x * newScale),
      y: Math.round(pointer.y - mousePointTo.y * newScale),
    };

    // Direct GPU/Canvas matrix update for immediate 60/120fps feedback
    stage.scale({ x: newScale, y: newScale });
    stage.position(newPos);
    stage.batchDraw();

    // Direct DOM text update for instant 120fps indicator feedback with zero React re-render lag
    const zoomIndicator = document.getElementById("stage-zoom-indicator");
    if (zoomIndicator) {
      zoomIndicator.textContent = `${Math.round(newScale * 100)}%`;
    }

    // Debounce React state synchronization until zooming finishes to eliminate re-rendering lag
    if (wheelRafRef.current !== null) {
      clearTimeout(wheelRafRef.current);
    }
    wheelRafRef.current = window.setTimeout(() => {
      setStageScale(newScale);
      setStagePos(newPos);
      wheelRafRef.current = null;
    }, 120);
  }, []);

  // Convert Stage pointer coordinates to natural image space
  const getStageRelativePos = useCallback((): [number, number] | null => {
    const stage = stageRef.current;
    if (!stage) return null;
    const pointer = stage.getPointerPosition();
    if (!pointer) return null;
    const transform = stage.getAbsoluteTransform().copy().invert();
    const pos = transform.point(pointer);
    if (pos.x < 0 || pos.x > imgDimensions.w || pos.y < 0 || pos.y > imgDimensions.h) {
      return null;
    }
    return [Math.round(pos.x), Math.round(pos.y)];
  }, [imgDimensions]);

  // Finish manual polygon
  const finishDrawingPolygon = useCallback(() => {
    if (drawPoints.length < 3) {
      setDrawPoints([]);
      setMouseCanvasPos(null);
      return;
    }

    const closedPoly = ensureClosedPolygon(drawPoints);
    const metrics = calculatePolygonArea(closedPoly, imgDimensions.w, imgDimensions.h);

    const nextId =
      segments.length > 0 ? Math.max(...segments.map((s) => s.id)) + 1 : 1;
    const colorHex = DISTINCT_COLORS[(nextId - 1) % DISTINCT_COLORS.length];

    const newSegment: EditableSegment = {
      id: nextId,
      id_str: `#${nextId}`,
      genus: formGenus,
      growth_form: formGrowthForm,
      condition: formCondition,
      color_hex: colorHex,
      polygon: closedPoly,
      area_px: metrics.area_px,
      area_pct: metrics.area_pct,
      visible: true,
      is_human_annotated: true,
    };

    addSegment(newSegment);
    setDrawPoints([]);
    setMouseCanvasPos(null);
    setToolMode("select");
    setEditorNotice(`Created coral polygon #${nextId}.`);
  }, [drawPoints, imgDimensions, segments, formGenus, formGrowthForm, formCondition, addSegment, setEditorNotice]);

  // Canvas Stage Click Handling
  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (toolMode === "select") {
      if (e.target === e.target.getStage() || e.target.name() === "bg-image") {
        setSelectedId(null);
      }
      return;
    }

    const pos = getStageRelativePos();
    if (!pos) return;
    const [clickX, clickY] = pos;

    if (toolMode === "sam") {
      const mouseEvt = e.evt as MouseEvent;
      const isNegativeModifier = mouseEvt.button === 2 || mouseEvt.altKey || mouseEvt.shiftKey;
      const pointLabel = isNegativeModifier ? 0 : samPointType;

      const isDuplicate = samPoints.some(
        (p) => Math.hypot(p.x - clickX, p.y - clickY) < 3
      );
      if (isDuplicate) return;

      const newPrompt: PromptPoint = {
        x: clickX,
        y: clickY,
        label: pointLabel,
      };

      setSamPoints((prev) => [...prev, newPrompt]);
      return;
    }

    if (toolMode === "polygon") {
      if (drawPoints.length >= 3) {
        const [startX, startY] = drawPoints[0];
        const dist = Math.hypot(clickX - startX, clickY - startY);
        if (dist < 18 / stageScale) {
          finishDrawingPolygon();
          return;
        }
      }
      setDrawPoints((prev) => [...prev, [clickX, clickY]]);
    }
  };

  const handleStageMouseMove = () => {
    if (toolMode === "polygon" && drawPoints.length > 0) {
      const pos = getStageRelativePos();
      setMouseCanvasPos(pos);
    }
  };

  // Cancel running SAM segmentation
  const handleCancelSam = useCallback(() => {
    if (samTimeoutRef.current !== null) {
      window.clearTimeout(samTimeoutRef.current);
      samTimeoutRef.current = null;
    }
    if (samAbortControllerRef.current) {
      samAbortControllerRef.current.abort("user_cancel");
      samAbortControllerRef.current = null;
    }
    setIsSamLoading(false);
    setEditorNotice("SAM boundary detection cancelled.");
  }, [setEditorNotice]);

  // Run SAM Model on currently placed points with timeout and abort controller
  const handleRunSam = useCallback(async () => {
    if (samPoints.length === 0 || isSamLoading) return;

    if (samTimeoutRef.current !== null) {
      window.clearTimeout(samTimeoutRef.current);
      samTimeoutRef.current = null;
    }
    if (samAbortControllerRef.current) {
      samAbortControllerRef.current.abort("restart");
    }

    const controller = new AbortController();
    samAbortControllerRef.current = controller;
    setIsSamLoading(true);

    // 20-second timeout guard
    samTimeoutRef.current = window.setTimeout(() => {
      if (samAbortControllerRef.current === controller) {
        controller.abort("timeout");
      }
    }, 20000);

    try {
      const res = await promptSegment(imageName, samPoints, imageSrc, controller.signal);
      if (res && res.polygon && res.polygon.length >= 3) {
        setSamPendingPolygon(res.polygon);
      } else {
        setEditorNotice("SAM was unable to trace a confident boundary. Place additional positive (+) points inside the coral.");
      }
    } catch (err: any) {
      if (controller.signal.aborted || err?.name === "AbortError") {
        const reason = (controller.signal as any).reason;
        if (reason === "timeout") {
          setEditorNotice("SAM boundary detection timed out (20s). Please retry with fewer points or draw manually.");
        } else {
          setEditorNotice("SAM boundary detection cancelled.");
        }
      } else {
        console.error("SAM Prompt segmentation error:", err);
        setEditorNotice(`SAM error: ${err?.message || err}`);
      }
    } finally {
      if (samTimeoutRef.current !== null) {
        window.clearTimeout(samTimeoutRef.current);
        samTimeoutRef.current = null;
      }
      if (samAbortControllerRef.current === controller) {
        samAbortControllerRef.current = null;
      }
      setIsSamLoading(false);
    }
  }, [imageName, samPoints, isSamLoading, imageSrc, setEditorNotice]);

  // Commit SAM mask as coral instance
  const handleCommitSamMask = useCallback(() => {
    if (!samPendingPolygon || samPendingPolygon.length < 3) return;
    const closed = ensureClosedPolygon(samPendingPolygon);
    const metrics = calculatePolygonArea(closed, imgDimensions.w, imgDimensions.h);

    const nextId =
      segments.length > 0 ? Math.max(...segments.map((s) => s.id)) + 1 : 1;
    const colorHex = DISTINCT_COLORS[(nextId - 1) % DISTINCT_COLORS.length];

    const newSegment: EditableSegment = {
      id: nextId,
      id_str: `#${nextId}`,
      genus: formGenus,
      growth_form: formGrowthForm,
      condition: formCondition,
      color_hex: colorHex,
      polygon: closed,
      area_px: metrics.area_px,
      area_pct: metrics.area_pct,
      visible: true,
      is_human_annotated: true,
    };

    addSegment(newSegment);
    setSamPoints([]);
    setSamPendingPolygon(null);
    setToolMode("select");
    setEditorNotice(`Added coral instance #${nextId}.`);
  }, [samPendingPolygon, imgDimensions, segments, formGenus, formGrowthForm, formCondition, addSegment, setEditorNotice]);

  // Tool selection handler
  const handleSelectTool = useCallback((mode: ToolMode) => {
    setToolMode(mode);
    if (mode === "select") {
      setSamPoints([]);
      setSamPendingPolygon(null);
      setDrawPoints([]);
      setSelectedVertexIdx(null);
    } else if (mode === "sam") {
      setDrawPoints([]);
      setSelectedVertexIdx(null);
    } else if (mode === "polygon") {
      setSamPoints([]);
      setSamPendingPolygon(null);
      setSelectedVertexIdx(null);
    }
  }, [setSelectedVertexIdx]);

  // Keyboard hotkeys
  useEditorHotkeys({
    isOpen,
    isSamLoading,
    toolMode,
    samPoints,
    samPendingPolygon,
    drawPoints,
    selectedId,
    selectedVertexIdx,
    hasPrev,
    hasNext,
    onExecuteSave: () => executeSave(true),
    onUndo: handleUndo,
    onRedo: redo,
    onRunSam: handleRunSam,
    onCommitSamMask: handleCommitSamMask,
    onUndoSamPoint: () => setSamPoints((prev) => prev.slice(0, -1)),
    onClearSamPoints: () => setSamPoints([]),
    onDiscardSamPending: () => setSamPendingPolygon(null),
    onFinishDrawingPolygon: finishDrawingPolygon,
    onUndoDrawPoint: () => setDrawPoints((prev) => prev.slice(0, -1)),
    onCancelDrawPolygon: () => {
      setDrawPoints([]);
      setMouseCanvasPos(null);
    },
    onSelectTool: handleSelectTool,
    onDeleteVertex: handleDeleteVertex,
    onDeleteSelected: handleDeleteSelected,
    onClearSelectedVertex: () => setSelectedVertexIdx(null),
    onDeselectCoral: () => setSelectedId(null),
    onFitToScreen: handleFitToScreen,
    onToggleShortcuts: () => setIsShortcutsOpen((prev) => !prev),
    onNavigatePrev,
    onNavigateNext,
    onClose,
    onCancelSam: handleCancelSam,
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-slate-900/60 backdrop-blur-xs select-none overflow-hidden animate-fade-in font-sans">
      {/* 1. Header Toolbar */}
      <EditorHeader
        imageName={imageName}
        imageResolution={imageResolution}
        imgDimensions={imgDimensions}
        saveStatus={saveStatus}
        hasUnsavedChanges={hasUnsavedChanges}
        lastSavedTime={lastSavedTime}
        toolMode={toolMode}
        isSamLoading={isSamLoading}
        canUndo={canUndo}
        canRedo={canRedo}
        hasPrev={hasPrev}
        hasNext={hasNext}
        isSaving={isSaving}
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
        onSelectTool={handleSelectTool}
        onUndo={handleUndo}
        onRedo={redo}
        onNavigatePrev={onNavigatePrev}
        onNavigateNext={onNavigateNext}
        onSave={() => executeSave(true)}
        onClose={onClose}
      />

      {/* 2. Main Studio Body: Viewport Canvas + Right Inspector */}
      <div className="flex flex-1 min-h-0 overflow-hidden relative">
        {/* Canvas Stage Viewport */}
        <div
          ref={containerRef}
          className="flex-1 min-w-0 h-full relative bg-[#0f172a] overflow-hidden select-none"
        >
          <EditorCanvas
            containerRef={containerRef}
            stageRef={stageRef}
            htmlImage={htmlImage}
            imgDimensions={imgDimensions}
            stageScale={stageScale}
            stagePos={stagePos}
            toolMode={toolMode}
            selectedId={selectedId}
            selectedVertexIdx={selectedVertexIdx}
            maskOpacity={maskOpacity}
            segments={segments}
            samPoints={samPoints}
            samPendingPolygon={samPendingPolygon}
            drawPoints={drawPoints}
            mouseCanvasPos={mouseCanvasPos}
            onStagePosChange={setStagePos}
            onWheel={handleWheel}
            onStageClick={handleStageClick}
            onStageMouseMove={handleStageMouseMove}
            onSelectCoral={(id) => {
              setSelectedId(id);
              setSelectedVertexIdx(null);
            }}
            onAddVertexOnEdge={handleAddVertexOnEdge}
            onSelectVertex={setSelectedVertexIdx}
            onVertexDrag={handleVertexDrag}
            onVertexDragEnd={handleVertexDragEnd}
            onDeleteVertex={handleDeleteVertex}
            isSamLoading={isSamLoading}
          />

          {/* Floating Canvas Controls (Zoom, Fit, Opacity) & Notices */}
          <EditorCanvasControls
            stageScale={stageScale}
            maskOpacity={maskOpacity}
            selectedId={selectedId}
            toolMode={toolMode}
            editorNotice={editorNotice}
            onZoomIn={() => handleZoomCenter(1.2)}
            onZoomOut={() => handleZoomCenter(0.83)}
            onFitToScreen={handleFitToScreen}
            onOpacityChange={setMaskOpacity}
            onOpenShortcuts={() => setIsShortcutsOpen(true)}
          />

          {/* Enhanced Multi-Point SAM Flow Top Banner */}
          {toolMode === "sam" && (
            <SamHudBanner
              samPoints={samPoints}
              samPendingPolygon={samPendingPolygon}
              samPointType={samPointType}
              isSamLoading={isSamLoading}
              coverPct={
                samPendingPolygon
                  ? calculatePolygonArea(samPendingPolygon, imgDimensions.w, imgDimensions.h).area_pct
                  : 0
              }
              onSetPointType={setSamPointType}
              onRunSam={handleRunSam}
              onCommitSamMask={handleCommitSamMask}
              onDiscardSamPending={() => {
                setSamPoints([]);
                setSamPendingPolygon(null);
              }}
              onUndoSamPoint={() => setSamPoints((prev) => prev.slice(0, -1))}
              onClearSamPoints={() => setSamPoints([])}
              onExitSam={() => {
                setToolMode("select");
                setSamPoints([]);
                setSamPendingPolygon(null);
              }}
              onRequirePositivePinNotice={() =>
                setEditorNotice(
                  "Please click on the coral to place at least one foreground (+) pin before running the SAM model."
                )
              }
              onCancelSam={handleCancelSam}
            />
          )}

          {/* Polygon Drawing In Progress Instruction Banner */}
          {toolMode === "polygon" && drawPoints.length > 0 && (
            <PolygonHudBanner
              onClosePolygon={finishDrawingPolygon}
              onCancel={() => {
                setDrawPoints([]);
                setMouseCanvasPos(null);
              }}
            />
          )}
        </div>

        {/* 3. Right Inspector & Corals Catalog Sidebar */}
        <EditorSidebar
          segments={segments}
          selectedId={selectedId}
          formGenus={formGenus}
          formGrowthForm={formGrowthForm}
          formCondition={formCondition}
          onSelectCoral={setSelectedId}
          onDeselectCoral={() => setSelectedId(null)}
          onDeleteSelected={handleDeleteSelected}
          onDeleteCoral={handleDeleteCoral}
          onToggleVisibility={handleToggleVisibility}
          onAddCoralClick={() => {
            setSelectedId(null);
            setToolMode("sam");
          }}
          onUpdateMetadata={handleUpdateSelectedMetadata}
          onSetFormGenus={setFormGenus}
          onSetFormGrowthForm={setFormGrowthForm}
          onSetFormCondition={setFormCondition}
        />
      </div>

      {/* 4. Fullscreen Blocking Overlay when SAM Inference is Running */}
      <SamLoadingOverlay
        isLoading={isSamLoading}
        pointsCount={samPoints.length}
        onCancel={handleCancelSam}
      />

      {/* 5. Keyboard Shortcuts & Controls Reference Dialog (? / /) */}
      <EditorShortcutsDialog
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />
    </div>
  );
};
