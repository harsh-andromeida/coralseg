import React from "react";
import { Wand2, Check, Sparkles, Undo2, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PromptPoint } from "../../types";

interface SamHudBannerProps {
  samPoints: PromptPoint[];
  samPendingPolygon: [number, number][] | null;
  samPointType: 1 | 0;
  isSamLoading: boolean;
  coverPct: number;
  onSetPointType: (type: 1 | 0) => void;
  onRunSam: () => void;
  onCommitSamMask: () => void;
  onDiscardSamPending: () => void;
  onUndoSamPoint: () => void;
  onClearSamPoints: () => void;
  onExitSam: () => void;
  onRequirePositivePinNotice: () => void;
}

export const SamHudBanner: React.FC<SamHudBannerProps> = ({
  samPoints,
  samPendingPolygon,
  samPointType,
  isSamLoading,
  coverPct,
  onSetPointType,
  onRunSam,
  onCommitSamMask,
  onDiscardSamPending,
  onUndoSamPoint,
  onClearSamPoints,
  onExitSam,
  onRequirePositivePinNotice,
}) => {
  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 flex items-center gap-3 bg-white/95 text-slate-900 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-200 shadow-xl animate-in fade-in slide-in-from-top-2 whitespace-nowrap select-none max-w-[95vw] overflow-x-auto">
      {/* Left: Mode Title & Badge */}
      <div className="flex items-center gap-2 shrink-0">
        <div className="size-7 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center shrink-0">
          <Wand2 size={15} />
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="text-xs font-bold text-slate-900 tracking-tight">SAM Smart Click</span>
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-teal-50 text-teal-700 border-teal-200 font-medium">
            Interactive
          </Badge>
        </div>
      </div>

      <div className="h-4 w-px bg-slate-200 shrink-0" />

      {/* Point Type Toggle (+ Coral / − Background) */}
      {!samPendingPolygon && (
        <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200 shrink-0">
          <button
            type="button"
            onClick={() => onSetPointType(1)}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-semibold cursor-pointer transition-all ${
              samPointType === 1
                ? "bg-emerald-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-white/80"
            }`}
            title="Add Positive Foreground Points (+)"
          >
            <span className="font-bold">+</span>
            <span>Coral</span>
          </button>
          <button
            type="button"
            onClick={() => onSetPointType(0)}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-semibold cursor-pointer transition-all ${
              samPointType === 0
                ? "bg-red-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-white/80"
            }`}
            title="Add Negative Background Points (−, or hold Alt/Shift)"
          >
            <span className="font-bold">−</span>
            <span>Background</span>
          </button>
        </div>
      )}

      {/* Points Count / Guidance Text */}
      <div className="text-xs text-slate-600 shrink-0">
        {samPendingPolygon ? (
          <span className="text-teal-700 font-semibold">
            Mask ready ({coverPct}% cover)
          </span>
        ) : samPoints.length === 0 ? (
          <span className="text-slate-500">
            Click coral to drop pin
          </span>
        ) : (
          <div className="flex items-center gap-1">
            <span className="font-semibold text-slate-800">
              {samPoints.length} pt{samPoints.length > 1 ? "s" : ""}
            </span>
            <span className="text-emerald-700 font-bold">
              ({samPoints.filter((p) => p.label === 1).length}+)
            </span>
            {samPoints.filter((p) => p.label === 0).length > 0 && (
              <span className="text-red-700 font-bold">
                ({samPoints.filter((p) => p.label === 0).length}−)
              </span>
            )}
          </div>
        )}
      </div>

      <div className="h-4 w-px bg-slate-200 shrink-0" />

      {/* Action Buttons */}
      <div className="flex items-center gap-1.5 shrink-0">
        {samPendingPolygon ? (
          <>
            <Button
              type="button"
              variant="default"
              size="sm"
              onClick={onCommitSamMask}
              className="h-7 px-3 text-xs font-bold bg-[#0d7c85] hover:bg-[#0b6b73] text-white shadow-xs cursor-pointer gap-1"
            >
              <Check size={13} />
              <span>Accept Mask</span>
              <kbd className="ml-1 text-[10px] bg-white/20 px-1 py-0.2 rounded font-mono">↵</kbd>
            </Button>

            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={isSamLoading}
              onClick={onRunSam}
              className="h-7 px-2.5 text-xs text-slate-700 hover:text-slate-900 border-slate-300 hover:bg-slate-50 cursor-pointer gap-1"
            >
              {isSamLoading ? <Loader2 size={12} className="animate-spin" /> : <Sparkles size={12} />}
              <span>Re-run</span>
              <kbd className="ml-1 text-[10px] bg-slate-200 px-1 py-0.2 rounded font-mono">Space</kbd>
            </Button>

            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onDiscardSamPending}
              className="h-7 px-2 text-xs text-slate-500 hover:text-slate-800"
            >
              Discard
            </Button>
          </>
        ) : (
          <>
            <Button
              type="button"
              variant="default"
              size="sm"
              disabled={isSamLoading}
              onClick={() => {
                if (samPoints.length === 0) {
                  onRequirePositivePinNotice();
                  return;
                }
                onRunSam();
              }}
              className={`h-7 px-3 text-xs font-bold text-white shadow-xs cursor-pointer gap-1.5 transition-all ${
                samPoints.length > 0
                  ? "bg-[#0d7c85] hover:bg-[#0b6b73]"
                  : "bg-[#0d7c85]/70 hover:bg-[#0d7c85]"
              }`}
            >
              {isSamLoading ? (
                <>
                  <Loader2 size={13} className="animate-spin" />
                  <span>Computing...</span>
                </>
              ) : (
                <>
                  <Sparkles size={13} />
                  <span>Run SAM Model</span>
                  <kbd className="ml-1 text-[10px] bg-white/20 px-1 py-0.2 rounded font-mono">↵</kbd>
                </>
              )}
            </Button>

            {samPoints.length > 0 && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={onUndoSamPoint}
                className="h-7 px-2 text-xs text-slate-600 hover:text-slate-900 border-slate-300"
                title="Undo last point (Backspace / ⌘Z)"
              >
                <Undo2 size={12} />
              </Button>
            )}

            {samPoints.length > 0 && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onClearSamPoints}
                className="h-7 px-2 text-xs text-slate-500 hover:text-slate-800"
              >
                Clear
              </Button>
            )}
          </>
        )}

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onExitSam}
          className="h-7 w-7 text-slate-400 hover:text-slate-700 ml-1"
          title="Exit SAM Tool (V)"
        >
          <X size={14} />
        </Button>
      </div>
    </div>
  );
};
