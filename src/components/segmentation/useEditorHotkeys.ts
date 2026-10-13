import { useEffect } from "react";
import { ToolMode } from "./types";
import { PromptPoint } from "../../types";

interface UseEditorHotkeysProps {
  isOpen: boolean;
  isSamLoading: boolean;
  toolMode: ToolMode;
  samPoints: PromptPoint[];
  samPendingPolygon: [number, number][] | null;
  drawPoints: [number, number][];
  selectedId: number | null;
  selectedVertexIdx: number | null;
  hasPrev?: boolean;
  hasNext?: boolean;
  onExecuteSave: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onRunSam: () => void;
  onCommitSamMask: () => void;
  onUndoSamPoint: () => void;
  onClearSamPoints: () => void;
  onDiscardSamPending: () => void;
  onFinishDrawingPolygon: () => void;
  onUndoDrawPoint: () => void;
  onCancelDrawPolygon: () => void;
  onSelectTool: (mode: ToolMode) => void;
  onDeleteVertex: (segId: number, vIdx: number) => void;
  onDeleteSelected: () => void;
  onClearSelectedVertex: () => void;
  onDeselectCoral: () => void;
  onFitToScreen: () => void;
  onNavigatePrev?: () => void;
  onNavigateNext?: () => void;
  onClose: () => void;
}

export function useEditorHotkeys({
  isOpen,
  isSamLoading,
  toolMode,
  samPoints,
  samPendingPolygon,
  drawPoints,
  selectedId,
  selectedVertexIdx,
  hasPrev = false,
  hasNext = false,
  onExecuteSave,
  onUndo,
  onRedo,
  onRunSam,
  onCommitSamMask,
  onUndoSamPoint,
  onClearSamPoints,
  onDiscardSamPending,
  onFinishDrawingPolygon,
  onUndoDrawPoint,
  onCancelDrawPolygon,
  onSelectTool,
  onDeleteVertex,
  onDeleteSelected,
  onClearSelectedVertex,
  onDeselectCoral,
  onFitToScreen,
  onNavigatePrev,
  onNavigateNext,
  onClose,
}: UseEditorHotkeysProps) {
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Pause all keyboard interaction while SAM model is running
      if (isSamLoading) {
        e.preventDefault();
        return;
      }

      const activeEl = document.activeElement as HTMLElement;
      const isInput =
        activeEl && ["INPUT", "TEXTAREA", "SELECT"].includes(activeEl.tagName);

      const key = e.key.toLowerCase();
      const isModifier = e.ctrlKey || e.metaKey;

      // Allow Ctrl+S / Cmd+S even inside input fields
      if (isModifier && key === "s") {
        e.preventDefault();
        onExecuteSave();
        return;
      }

      // If user is actively typing in a form input, avoid intercepting single-letter hotkeys
      if (isInput) return;

      // 1. Undo: Ctrl+Z or Cmd+Z
      if (isModifier && key === "z" && !e.shiftKey) {
        e.preventDefault();
        onUndo();
        return;
      }

      // 2. Redo: Cmd+Shift+Z, Ctrl+Shift+Z, or Ctrl+Y / Cmd+Y
      if ((isModifier && e.shiftKey && key === "z") || (isModifier && key === "y")) {
        e.preventDefault();
        onRedo();
        return;
      }

      // 3. SAM specific flow shortcuts
      if (toolMode === "sam") {
        if (e.key === "Enter") {
          e.preventDefault();
          if (samPendingPolygon) {
            onCommitSamMask();
          } else if (samPoints.length > 0) {
            onRunSam();
          }
          return;
        }

        if (e.key === " " && !samPendingPolygon && samPoints.length > 0) {
          e.preventDefault();
          onRunSam();
          return;
        }

        if ((key === "backspace" || key === "delete") && !samPendingPolygon && samPoints.length > 0) {
          e.preventDefault();
          onUndoSamPoint();
          return;
        }
      }

      // 4. Manual Polygon shortcuts
      if (toolMode === "polygon" && drawPoints.length > 0) {
        if (e.key === "Enter" && drawPoints.length >= 3) {
          e.preventDefault();
          onFinishDrawingPolygon();
          return;
        }
        if (key === "backspace" || key === "delete") {
          e.preventDefault();
          onUndoDrawPoint();
          return;
        }
      }

      // 5. Tool switches
      if (key === "v") {
        onSelectTool("select");
      } else if (key === "w") {
        onSelectTool("sam");
      } else if (key === "p") {
        onSelectTool("polygon");
      } else if (key === "delete" || key === "backspace") {
        // If an individual vertex is selected, delete only that vertex
        if (selectedVertexIdx !== null && selectedId !== null) {
          e.preventDefault();
          onDeleteVertex(selectedId, selectedVertexIdx);
          return;
        }
        // Otherwise, delete the entire selected coral annotation
        if (selectedId !== null) {
          e.preventDefault();
          onDeleteSelected();
          return;
        }
      } else if (e.key === "Escape") {
        if (selectedVertexIdx !== null) {
          e.preventDefault();
          onClearSelectedVertex();
          return;
        }
        if (toolMode === "polygon" && drawPoints.length > 0) {
          onCancelDrawPolygon();
        } else if (toolMode === "sam" && samPendingPolygon) {
          onDiscardSamPending();
        } else if (toolMode === "sam" && samPoints.length > 0) {
          onClearSamPoints();
        } else if (selectedId !== null) {
          onDeselectCoral();
        } else {
          onClose();
        }
      } else if (key === "0" || key === "f") {
        onFitToScreen();
      } else if (key === "[" && hasPrev && onNavigatePrev) {
        onNavigatePrev();
      } else if (key === "]" && hasNext && onNavigateNext) {
        onNavigateNext();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
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
    onExecuteSave,
    onUndo,
    onRedo,
    onRunSam,
    onCommitSamMask,
    onUndoSamPoint,
    onClearSamPoints,
    onDiscardSamPending,
    onFinishDrawingPolygon,
    onUndoDrawPoint,
    onCancelDrawPolygon,
    onSelectTool,
    onDeleteVertex,
    onDeleteSelected,
    onClearSelectedVertex,
    onDeselectCoral,
    onFitToScreen,
    onNavigatePrev,
    onNavigateNext,
    onClose,
  ]);
}
