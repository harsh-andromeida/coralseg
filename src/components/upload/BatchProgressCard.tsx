import React from "react";
import {
  Rocket,
  CheckCircle2,
  Play,
  Pause,
  StopCircle,
  Loader2,
  Cpu,
  ArrowRight,
  FolderArchive,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { BatchStatusResponse } from "../../types";

export interface BatchProgressCardProps {
  itemsCount: number;
  onStartBatch: () => void;
  isStarting: boolean;
  batchStatus: BatchStatusResponse | null;
  isBackendConnected: boolean;
  exportFeedback: string | null;
  onClearExportFeedback: () => void;
  isCancellingBatch: boolean;
  onPauseResumeBatch: () => void;
  onCancelBatch: () => void;
  onLaunchStudioNow: () => void;
  onOpenExportModal: () => void;
}

export const BatchProgressCard: React.FC<BatchProgressCardProps> = ({
  itemsCount,
  onStartBatch,
  isStarting,
  batchStatus,
  isBackendConnected,
  exportFeedback,
  onClearExportFeedback,
  isCancellingBatch,
  onPauseResumeBatch,
  onCancelBatch,
  onLaunchStudioNow,
  onOpenExportModal,
}) => {
  const isRunning = batchStatus?.is_running ?? false;
  const isPaused = batchStatus?.is_paused ?? false;
  const isComplete =
    !isRunning &&
    (batchStatus?.completed ?? 0) > 0 &&
    batchStatus?.completed === batchStatus?.total;
  const hasProcessed = (batchStatus?.completed ?? 0) > 0;

  if (!isRunning && !isPaused && !isComplete && !hasProcessed) {
    return (
      <div className="shrink-0 pt-2.5 mt-auto border-t border-slate-100 bg-white">
        <Button
          type="button"
          className="w-full h-10 text-xs font-bold tracking-wider uppercase text-white rounded-xl shadow-xs hover:shadow-md transition-all gap-2 bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] hover:opacity-95 active:scale-[0.99]"
          onClick={onStartBatch}
          disabled={isStarting || itemsCount === 0 || !isBackendConnected}
        >
          <Rocket size={14} className={isStarting ? "animate-bounce" : ""} />
          <span>
            {isStarting
              ? "Initializing Background Engines..."
              : !isBackendConnected
              ? "Waiting for Vision Engine..."
              : `Start Batch Processing (${itemsCount} Images)`}
          </span>
        </Button>
      </div>
    );
  }

  return (
    <div className="shrink-0 pt-2.5 mt-auto border-t border-slate-200/90 bg-white">
      {exportFeedback && (
        <div className="mb-2 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between">
          <span>{exportFeedback}</span>
          <button
            onClick={onClearExportFeedback}
            className="text-emerald-600 hover:text-emerald-900"
          >
            <X size={12} />
          </button>
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3 shadow-xs space-y-2.5">
        {/* Row 1: Header status and controls */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            {isRunning && !isPaused && (
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-teal-600" />
              </span>
            )}
            {isPaused && <span className="h-2.5 w-2.5 rounded-full bg-amber-500" />}
            {isComplete && <CheckCircle2 size={14} className="text-emerald-600 shrink-0" />}

            <span className="text-xs font-bold text-slate-800 truncate">
              {isComplete
                ? "Batch Complete"
                : isPaused
                ? "Batch Paused"
                : isRunning
                ? "Autonomous Background Processing"
                : "Batch Idle"}
            </span>

            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-600">
              {batchStatus?.completed || 0} / {batchStatus?.total || itemsCount} Done
            </span>
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {(isRunning || isPaused) && (
              <>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={onPauseResumeBatch}
                  disabled={isCancellingBatch}
                  className="h-7 px-2 text-[10px] font-semibold gap-1 text-slate-700 hover:bg-slate-100"
                  title={isPaused ? "Resume processing" : "Pause processing"}
                >
                  {isPaused ? <Play size={11} /> : <Pause size={11} />}
                  <span>{isPaused ? "Resume" : "Pause"}</span>
                </Button>

                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={onCancelBatch}
                  disabled={isCancellingBatch}
                  className="h-7 px-2 text-[10px] font-semibold gap-1 text-red-600 hover:bg-red-50 hover:border-red-200 border-red-200"
                  title="Cancel processing and unload models from RAM"
                >
                  {isCancellingBatch ? (
                    <Loader2 size={11} className="animate-spin text-red-600" />
                  ) : (
                    <StopCircle size={11} />
                  )}
                  <span>{isCancellingBatch ? "Unloading..." : "Cancel"}</span>
                </Button>
              </>
            )}

            {!isRunning && !isPaused && hasProcessed && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={onStartBatch}
                className="h-7 px-2 text-[10px] font-semibold gap-1 text-teal-700 hover:bg-teal-50"
                title="Re-run batch on selected images"
              >
                <Rocket size={11} />
                <span>Restart</span>
              </Button>
            )}
          </div>
        </div>

        {/* Row 2: Progress bar & Live stage readout with RAM model loading indicator */}
        <div className="space-y-1">
          <div className="flex justify-between items-center text-[11px] font-medium text-slate-600">
            <div className="flex items-center gap-1.5 min-w-0 pr-2">
              {batchStatus?.current_stage?.includes("Loading") && (
                <span className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-900 border border-amber-300 shrink-0 animate-pulse">
                  <Cpu size={10} className="text-amber-700" />
                  <span>Loading Model into RAM</span>
                </span>
              )}
              <span className="truncate max-w-[340px]">
                {isRunning && !isPaused
                  ? batchStatus?.current_stage?.includes("Loading")
                    ? batchStatus.current_stage
                    : batchStatus?.current_image
                    ? `Analyzing ${batchStatus.current_image} • ${
                        batchStatus.current_stage || "Processing"
                      }`
                    : batchStatus?.current_stage || "Processing..."
                  : isPaused
                  ? "Paused (Current progress preserved)"
                  : batchStatus?.current_stage?.includes("Cancelled") ||
                    batchStatus?.current_stage === "cancelled"
                  ? "Cancelled • Models unloaded from RAM"
                  : isComplete
                  ? "All images segmented and taxonomically enriched"
                  : "Ready for analysis"}
              </span>
            </div>
            <span className="font-mono text-teal-700 font-bold shrink-0">
              {batchStatus?.percent || 0}%
            </span>
          </div>
          <Progress
            value={batchStatus?.percent || 0}
            className="h-2 bg-slate-200/80 rounded-full"
            indicatorColor={
              batchStatus?.current_stage?.includes("Loading")
                ? "bg-gradient-to-r from-amber-500 via-orange-500 to-amber-600 animate-pulse"
                : "bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85]"
            }
          />
        </div>

        {/* Row 3: Action Buttons (View Progress & Download COCO Dataset) */}
        <div className="flex items-center gap-2 pt-1">
          <Button
            type="button"
            className="flex-1 h-9 text-xs font-bold uppercase tracking-wider text-white rounded-lg shadow-xs hover:shadow-md transition-all gap-1.5 bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] hover:opacity-95"
            onClick={onLaunchStudioNow}
            disabled={!isBackendConnected || (!hasProcessed && !isRunning)}
          >
            <span>
              {hasProcessed
                ? `Open Studio (${batchStatus?.completed} Ready)`
                : "View Progress in Studio"}
            </span>
            <ArrowRight size={13} />
          </Button>

          <Button
            type="button"
            variant="outline"
            className="h-9 px-3 text-xs font-bold gap-1.5 border-teal-300 text-teal-800 bg-teal-50/60 hover:bg-teal-100 hover:border-teal-400 rounded-lg transition-colors cursor-pointer"
            onClick={onOpenExportModal}
            disabled={!hasProcessed}
            title="Export complete structured dataset into a custom destination folder or as a .zip"
          >
            <FolderArchive size={14} className="text-teal-700" />
            <span>Export Dataset</span>
          </Button>
        </div>
      </div>
    </div>
  );
};
