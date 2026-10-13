import React from "react";
import { Sparkles } from "lucide-react";

interface SamLoadingOverlayProps {
  isLoading: boolean;
  pointsCount: number;
}

export const SamLoadingOverlay: React.FC<SamLoadingOverlayProps> = ({
  isLoading,
  pointsCount,
}) => {
  if (!isLoading) return null;

  return (
    <div
      className="fixed inset-0 z-[100] bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center text-white select-none cursor-wait pointer-events-auto animate-in fade-in duration-200"
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      onContextMenu={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
    >
      <div className="bg-slate-900/95 border border-teal-500/40 rounded-2xl p-7 shadow-2xl flex flex-col items-center max-w-sm text-center mx-4 animate-in fade-in zoom-in-95 duration-200">
        <div className="relative mb-5 flex items-center justify-center">
          <div className="w-16 h-16 rounded-full border-4 border-teal-500/20 border-t-teal-400 animate-spin" />
          <Sparkles className="w-7 h-7 text-teal-300 absolute inset-0 m-auto animate-pulse" />
        </div>
        <h3 className="text-base font-bold text-white mb-1.5 tracking-tight">
          Running SAM Model...
        </h3>
        <p className="text-xs text-slate-300 mb-4 leading-relaxed">
          Neural foundation inference is computing segmentation mask on CPU. All interactions are paused.
        </p>
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-teal-500/10 border border-teal-500/30 text-xs font-medium text-teal-300">
          <span className="w-2 h-2 rounded-full bg-teal-400 animate-ping" />
          <span>
            Processing {pointsCount} prompt {pointsCount === 1 ? "point" : "points"}...
          </span>
        </div>
      </div>
    </div>
  );
};
