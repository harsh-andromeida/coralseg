import React from "react";
import {
  X,
  Save,
  Undo2,
  Redo2,
  MousePointer,
  Wand2,
  Pentagon,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Loader2,
  HelpCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ToolMode, SaveStatus } from "./types";

interface EditorHeaderProps {
  imageName: string;
  imageResolution?: string;
  imgDimensions: { w: number; h: number };
  saveStatus: SaveStatus;
  hasUnsavedChanges: boolean;
  lastSavedTime: string | null;
  toolMode: ToolMode;
  isSamLoading: boolean;
  canUndo: boolean;
  canRedo: boolean;
  hasPrev?: boolean;
  hasNext?: boolean;
  isSaving: boolean;
  onOpenShortcuts?: () => void;
  onSelectTool: (mode: ToolMode) => void;
  onUndo: () => void;
  onRedo: () => void;
  onNavigatePrev?: () => void;
  onNavigateNext?: () => void;
  onSave: () => void;
  onClose: () => void;
}

export const EditorHeader: React.FC<EditorHeaderProps> = ({
  imageName,
  imageResolution,
  imgDimensions,
  saveStatus,
  hasUnsavedChanges,
  lastSavedTime,
  toolMode,
  isSamLoading,
  canUndo,
  canRedo,
  hasPrev = false,
  hasNext = false,
  isSaving,
  onOpenShortcuts,
  onSelectTool,
  onUndo,
  onRedo,
  onNavigatePrev,
  onNavigateNext,
  onSave,
  onClose,
}) => {
  return (
    <header className="h-14 bg-white border-b border-slate-200 px-4 flex items-center justify-between gap-4 shrink-0 shadow-xs z-20">
      {/* Left: Branding, Filename, Resolution, Save Status */}
      <div className="flex items-center gap-3 min-w-0">
        <Badge
          variant="outline"
          className="text-[10px] font-mono font-bold uppercase tracking-wider bg-teal-50 text-teal-800 border-teal-200 px-2 py-0.5 shrink-0"
        >
          Human Annotation
        </Badge>

        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-semibold text-slate-900 truncate max-w-[200px] md:max-w-[320px]">
            {imageName}
          </span>
          <span className="text-[11px] font-mono text-slate-500 shrink-0">
            {imageResolution && imageResolution !== "0x0"
              ? imageResolution
              : `${imgDimensions.w}x${imgDimensions.h}`}
          </span>
        </div>

        {/* Live Autosave & Save Status Indicator */}
        <div className="hidden sm:flex items-center shrink-0">
          {saveStatus === "saving" ? (
            <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border bg-blue-50 border-blue-200 text-blue-800">
              <Loader2 size={12} className="animate-spin text-blue-600" />
              <span>Autosaving...</span>
            </div>
          ) : hasUnsavedChanges || saveStatus === "unsaved" ? (
            <div
              className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border bg-amber-50 border-amber-200 text-amber-800"
              title="Edits made — will auto-save in 20s or press Cmd+S"
            >
              <span className="size-2 rounded-full bg-amber-500 animate-pulse" />
              <span>Unsaved (auto-saves in 20s)</span>
            </div>
          ) : (
            <div
              className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border bg-emerald-50 border-emerald-200 text-emerald-800"
              title="Autosaves every 20 seconds"
            >
              <CheckCircle2 size={12} className="text-emerald-600" />
              <span>
                {lastSavedTime ? `Auto-saved at ${lastSavedTime}` : "Auto-save active (20s)"}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Center: Tools Selector (CVAT / Roboflow Style) */}
      <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
        <button
          type="button"
          onClick={() => onSelectTool("select")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
            toolMode === "select"
              ? "bg-[#0d7c85] text-white shadow-xs"
              : "text-slate-600 hover:text-slate-900 hover:bg-white"
          }`}
          title="Select & Edit Vertices (Hotkey: V)"
        >
          <MousePointer size={14} />
          <span>Select (V)</span>
        </button>

        <button
          type="button"
          onClick={() => onSelectTool("sam")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
            toolMode === "sam"
              ? "bg-[#0d7c85] text-white shadow-xs"
              : "text-slate-600 hover:text-slate-900 hover:bg-white"
          }`}
          title="SAM Smart Click Model (Hotkey: W)"
        >
          {isSamLoading ? (
            <Loader2 size={14} className="animate-spin text-teal-200" />
          ) : (
            <Wand2 size={14} />
          )}
          <span>SAM Smart Click (W)</span>
        </button>

        <button
          type="button"
          onClick={() => onSelectTool("polygon")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
            toolMode === "polygon"
              ? "bg-[#0d7c85] text-white shadow-xs"
              : "text-slate-600 hover:text-slate-900 hover:bg-white"
          }`}
          title="Manual Polygon Boundary Draw (Hotkey: P)"
        >
          <Pentagon size={14} />
          <span>Polygon (P)</span>
        </button>
      </div>

      {/* Right: Undo/Redo, Navigation, Save & Close */}
      <div className="flex items-center gap-2">
        {/* Undo / Redo */}
        <div className="hidden md:flex items-center gap-1 border-r border-slate-200 pr-2">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            disabled={!canUndo}
            onClick={onUndo}
            className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
            title="Undo (Ctrl+Z / Cmd+Z)"
          >
            <Undo2 size={15} />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            disabled={!canRedo}
            onClick={onRedo}
            className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
            title="Redo (Ctrl+Y / Cmd+Shift+Z)"
          >
            <Redo2 size={15} />
          </Button>
        </div>

        {/* Quick Pagination */}
        {(hasPrev || hasNext) && (
          <div className="flex items-center gap-1 border-r border-slate-200 pr-2">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              disabled={!hasPrev}
              onClick={onNavigatePrev}
              className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
              title="Previous Image ([)"
            >
              <ChevronLeft size={16} />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              disabled={!hasNext}
              onClick={onNavigateNext}
              className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
              title="Next Image (])"
            >
              <ChevronRight size={16} />
            </Button>
          </div>
        )}

        {/* Save Button */}
        <Button
          type="button"
          variant="default"
          size="sm"
          disabled={isSaving}
          onClick={onSave}
          className="h-8 px-3 text-xs font-semibold gap-1.5 bg-[#0d7c85] hover:bg-[#0b6b73] text-white shadow-xs cursor-pointer"
          title="Save Annotations (Ctrl+S / Cmd+S)"
        >
          {isSaving ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
          <span>Save</span>
          <kbd className="hidden md:inline-block ml-0.5 text-[10px] bg-white/20 px-1 py-0.2 rounded font-mono">
            ⌘S
          </kbd>
        </Button>

        {/* Shortcuts / Controls Help (?) */}
        {onOpenShortcuts && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onOpenShortcuts}
            className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            title="Controls & Shortcuts (? / /)"
          >
            <HelpCircle size={16} />
          </Button>
        )}

        {/* Close */}
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 text-slate-500 hover:text-slate-900 hover:bg-slate-100 ml-1"
          title="Close Editor (Esc)"
        >
          <X size={18} />
        </Button>
      </div>
    </header>
  );
};
