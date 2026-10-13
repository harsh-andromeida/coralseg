import React from "react";
import { ZoomIn, ZoomOut, RotateCcw, Sliders, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ToolMode } from "./types";

interface EditorCanvasControlsProps {
  stageScale: number;
  maskOpacity: number;
  selectedId: number | null;
  toolMode: ToolMode;
  editorNotice: string | null;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToScreen: () => void;
  onOpacityChange: (opacity: number) => void;
}

export const EditorCanvasControls: React.FC<EditorCanvasControlsProps> = ({
  stageScale,
  maskOpacity,
  selectedId,
  toolMode,
  editorNotice,
  onZoomIn,
  onZoomOut,
  onFitToScreen,
  onOpacityChange,
}) => {
  return (
    <>
      {/* Floating Canvas Controls (Zoom, Fit, Opacity) */}
      <div className="absolute bottom-4 left-4 z-10 flex items-center gap-2 bg-white/95 backdrop-blur-md p-1.5 rounded-lg border border-slate-200 shadow-md text-xs text-slate-700">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onZoomOut}
          className="h-7 w-7 text-slate-600 hover:text-slate-900"
          title="Zoom Out (-)"
        >
          <ZoomOut size={13} />
        </Button>

        <span className="font-mono text-[11px] text-slate-700 px-1 min-w-[42px] text-center font-semibold">
          {Math.round(stageScale * 100)}%
        </span>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onZoomIn}
          className="h-7 w-7 text-slate-600 hover:text-slate-900"
          title="Zoom In (+)"
        >
          <ZoomIn size={13} />
        </Button>

        <div className="h-4 w-px bg-slate-200 mx-1" />

        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onFitToScreen}
          className="h-7 px-2 text-[11px] text-slate-600 hover:text-slate-900 gap-1"
          title="Fit to Screen (F / 0)"
        >
          <RotateCcw size={12} />
          <span>Fit (F)</span>
        </Button>

        <div className="h-4 w-px bg-slate-200 mx-1" />

        {/* Opacity Slider */}
        <div className="flex items-center gap-1.5 px-2 text-[11px] text-slate-600">
          <Sliders size={11} />
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={maskOpacity}
            onChange={(e) => onOpacityChange(parseFloat(e.target.value))}
            className="w-16 accent-teal-600 cursor-pointer"
            title="Mask Opacity"
          />
        </div>
      </div>

      {/* Floating Polygon Editing UX Guide Pill */}
      {selectedId !== null && toolMode === "select" && (
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-teal-500/40 shadow-lg text-[11px] text-slate-200">
          <div className="w-2 h-2 rounded-full bg-[#00e5ff] animate-pulse shrink-0" />
          <span className="font-semibold text-white">Edit Coral #{selectedId}:</span>
          <span className="text-slate-300">
            Drag points to move • Click edge or (+) handle to add point • Right-click (or Alt+click / Del) to remove point
          </span>
        </div>
      )}

      {/* Floating Notice Banner */}
      {editorNotice && (
        <div className="absolute top-14 left-4 z-20 flex items-center gap-2 bg-slate-900/95 text-teal-200 border border-teal-500/50 px-3 py-1.5 rounded-md shadow-xl text-xs animate-in fade-in duration-150">
          <Info size={13} className="text-teal-400 shrink-0" />
          <span>{editorNotice}</span>
        </div>
      )}
    </>
  );
};
