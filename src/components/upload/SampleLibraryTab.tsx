import React from "react";
import { Check, Loader2 } from "lucide-react";
import { UISelect } from "../UISelect";
import { Button } from "@/components/ui/button";
import { SampleItem, BatchStatusResponse } from "../../types";
import { getSampleImageUrl } from "../../services/api";
import { BatchProgressCard } from "./BatchProgressCard";

export interface SampleLibraryTabProps {
  availableSamples: SampleItem[];
  sampleSubset: SampleItem[];
  selectedPreset: string;
  onSelectPreset: (preset: string) => void;
  selectedSpecific: string[];
  onToggleSpecific: (filename: string, checked: boolean) => void;
  isBackendConnected: boolean;
  onManualRetry: () => void;
  onOpenRemoteModal: () => void;
  onStartBatch: () => void;
  isStartingBatch: boolean;
  batchStatus: BatchStatusResponse | null;
  exportFeedback: string | null;
  onClearExportFeedback: () => void;
  isCancellingBatch: boolean;
  onPauseResumeBatch: () => void;
  onCancelBatch: () => void;
  onLaunchStudioNow: () => void;
  onOpenExportModal: () => void;
}

export const SampleLibraryTab: React.FC<SampleLibraryTabProps> = ({
  availableSamples,
  sampleSubset,
  selectedPreset,
  onSelectPreset,
  selectedSpecific,
  onToggleSpecific,
  isBackendConnected,
  onManualRetry,
  onOpenRemoteModal,
  onStartBatch,
  isStartingBatch,
  batchStatus,
  exportFeedback,
  onClearExportFeedback,
  isCancellingBatch,
  onPauseResumeBatch,
  onCancelBatch,
  onLaunchStudioNow,
  onOpenExportModal,
}) => {
  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      {/* Single-line Toolbar Header */}
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 mb-2 pb-1.5 border-b border-slate-100">
        <div>
          <h4 className="text-sm font-bold text-[#0f1e4a] tracking-tight">
            Select from Sample Library
          </h4>
          <p className="text-[11px] text-slate-500">
            {sampleSubset.length} sample image{sampleSubset.length === 1 ? "" : "s"} selected for
            instance segmentation
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <label className="text-xs font-semibold text-slate-600">Batch:</label>
          <div className="w-48">
            <UISelect
              value={selectedPreset}
              onChange={onSelectPreset}
              options={[
                { value: "First 6 Samples", label: "First 6 Samples" },
                { value: "First 12 Samples", label: "First 12 Samples" },
                { value: "First 24 Samples", label: "First 24 Samples" },
                { value: "All Available Samples", label: "All Available Samples" },
                { value: "Select Specific Files", label: "Select Specific Files" },
              ]}
              placeholder="Select batch preset"
            />
          </div>
        </div>
      </div>

      {selectedPreset === "Select Specific Files" && (
        <div className="shrink-0 form-group mb-2">
          <label className="form-label text-xs font-semibold text-slate-600">Choose Files</label>
          <div className="max-h-28 overflow-y-auto border border-slate-200 rounded-lg p-2 bg-slate-50/50 space-y-1">
            {availableSamples.map((s) => (
              <label
                key={s.filename}
                className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer hover:bg-white p-1 rounded"
              >
                <input
                  type="checkbox"
                  className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                  checked={selectedSpecific.includes(s.filename)}
                  onChange={(e) => onToggleSpecific(s.filename, e.target.checked)}
                />
                <span className="font-mono text-[11px]">{s.filename}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Scrollable Imagery Container - Image Cards Grid */}
      <div className="portal-scroll-area flex-1 min-h-0">
        {!isBackendConnected ? (
          <div className="p-8 text-center bg-slate-50 rounded-xl border border-dashed border-slate-200 text-slate-500 text-xs my-4 flex flex-col items-center gap-2.5">
            <div className="flex items-center gap-2 font-semibold text-slate-700 text-sm">
              <Loader2 size={16} className="animate-spin text-teal-600" />
              <span>Connecting to Python Vision Engine...</span>
            </div>
            <p className="text-slate-500 max-w-md text-xs">
              Please ensure the backend is started locally (<code>./start_backend.sh</code>) or
              connect to your Google Colab / remote tunnel instance.
            </p>
            <div className="flex items-center gap-2 mt-1">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 text-xs border-slate-300"
                onClick={onManualRetry}
              >
                Retry Connection
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                className="h-7 text-xs bg-teal-50 text-teal-700 hover:bg-teal-100 border border-teal-200"
                onClick={onOpenRemoteModal}
              >
                Set Google Colab / Remote URL
              </Button>
            </div>
          </div>
        ) : sampleSubset.length === 0 ? (
          <div className="p-8 text-center bg-slate-50 rounded-xl border border-dashed border-slate-200 text-slate-500 text-xs my-4">
            No sample images found. Please verify <code>demo_images/</code> folder.
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 my-1">
            {sampleSubset.map((s, idx) => (
              <div
                key={idx}
                className="group relative rounded-xl border border-slate-200/90 bg-white overflow-hidden shadow-xs hover:shadow-md hover:border-teal-500/60 transition-all duration-200 flex flex-col"
              >
                <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-100">
                  <img
                    src={getSampleImageUrl(s.filename)}
                    alt={s.filename}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 ease-out"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'><rect width='100' height='100' fill='%23eee'/><text x='50%' y='50%' text-anchor='middle' fill='%23aaa' dy='.3em'>Image</text></svg>";
                    }}
                  />
                  {/* Floating index badge top-left */}
                  <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded-md bg-slate-900/70 backdrop-blur-md text-white font-mono text-[10px] font-semibold tracking-wider shadow-xs">
                    #{String(idx + 1).padStart(2, "0")}
                  </div>
                  {/* Selected indicator top-right */}
                  <div className="absolute top-2 right-2 h-5 w-5 rounded-full bg-emerald-500/90 text-white flex items-center justify-center shadow-xs">
                    <Check size={11} strokeWidth={3} />
                  </div>
                </div>

                {/* Clean metadata caption bar */}
                <div className="px-2 py-1.5 bg-white flex items-center justify-between gap-1 border-t border-slate-100">
                  <span
                    className="text-[11px] font-medium text-slate-700 truncate"
                    title={s.filename}
                  >
                    {s.filename}
                  </span>
                  <span className="text-[9px] font-mono uppercase text-slate-400 bg-slate-50 border border-slate-200/60 px-1 py-0.2 rounded shrink-0">
                    {s.filename.split(".").pop() || "PNG"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Fixed Bottom Launch Bar - Go Button ALWAYS visible */}
      <BatchProgressCard
        itemsCount={sampleSubset.length}
        onStartBatch={onStartBatch}
        isStarting={isStartingBatch}
        batchStatus={batchStatus}
        isBackendConnected={isBackendConnected}
        exportFeedback={exportFeedback}
        onClearExportFeedback={onClearExportFeedback}
        isCancellingBatch={isCancellingBatch}
        onPauseResumeBatch={onPauseResumeBatch}
        onCancelBatch={onCancelBatch}
        onLaunchStudioNow={onLaunchStudioNow}
        onOpenExportModal={onOpenExportModal}
      />
    </div>
  );
};
