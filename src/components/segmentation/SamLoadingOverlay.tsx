import React from "react";
import { Sparkles, X } from "lucide-react";

interface SamLoadingOverlayProps {
  isLoading: boolean;
  pointsCount: number;
  onCancel?: () => void;
}

export const SamLoadingOverlay: React.FC<SamLoadingOverlayProps> = ({
  isLoading,
  pointsCount,
  onCancel,
}) => {
  if (!isLoading) return null;

  return (
    <div
      tabIndex={0}
      className="fixed inset-0 z-[100] bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center text-white select-none cursor-wait pointer-events-auto animate-in fade-in duration-200 outline-none"
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      onContextMenu={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      onKeyDown={(e) => {
        e.stopPropagation();
        e.preventDefault();
        if (e.key === "Escape" && onCancel) {
          onCancel();
        }
      }}
    >
      <div className="bg-slate-900/95 border border-teal-500/40 rounded-2xl p-7 shadow-2xl flex flex-col items-center max-w-sm text-center mx-4 animate-in fade-in zoom-in-95 duration-200">
        <div className="relative mb-5 flex items-center justify-center">
          <div className="w-16 h-16 rounded-full border-4 border-teal-500/20 border-t-teal-400 animate-spin" />
          <Sparkles className="w-7 h-7 text-teal-300 absolute inset-0 m-auto animate-pulse" />
        </div>
        <h3 className="text-base font-bold text-white mb-1.5 tracking-tight">
          Detecting Coral Boundary...
        </h3>
        <p className="text-xs text-slate-300 mb-4 leading-relaxed">
          Tracing coral contour from prompt points. Please wait a moment...
        </p>
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-teal-500/10 border border-teal-500/30 text-xs font-medium text-teal-300">
          <span className="w-2 h-2 rounded-full bg-teal-400 animate-ping" />
          <span>
            Tracing {pointsCount} prompt {pointsCount === 1 ? "point" : "points"}...
          </span>
        </div>

        {onCancel && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onCancel();
            }}
            className="mt-5 px-3.5 py-1.5 rounded-lg bg-slate-800/90 hover:bg-slate-700/90 border border-slate-700 hover:border-slate-500 text-xs font-medium text-slate-300 hover:text-white transition-all shadow-sm cursor-pointer flex items-center gap-1.5"
            title="Cancel segmentation (Esc)"
          >
            <X className="w-3.5 h-3.5" />
            <span>Cancel</span>
            <kbd className="ml-1 text-[10px] bg-slate-700/80 px-1 py-0.5 rounded font-mono text-slate-400">Esc</kbd>
          </button>
        )}
      </div>
    </div>
  );
};
