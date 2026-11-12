import React, { useState, useRef } from "react";
import { UploadCloud, CheckCircle2, X } from "lucide-react";
import { LoadedImage, BatchStatusResponse } from "../../types";
import { BatchProgressCard } from "./BatchProgressCard";

export interface ImageUploadTabProps {
  stagedImages: LoadedImage[];
  onFilesSelected: (files: FileList) => void;
  onRemoveImage: (index: number) => void;
  onClearAll: () => void;
  onStartBatch: () => void;
  isStartingBatch: boolean;
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

export const ImageUploadTab: React.FC<ImageUploadTabProps> = ({
  stagedImages,
  onFilesSelected,
  onRemoveImage,
  onClearAll,
  onStartBatch,
  isStartingBatch,
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
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const onDragLeave = () => setDragActive(false);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      <div className="shrink-0 mb-2">
        <h4 className="text-sm font-bold text-[#0f1e4a] tracking-tight">
          Upload Coral Imagery
        </h4>
        <p className="text-[11px] text-slate-500">
          Import underwater survey photos or a .zip archive for batch instance segmentation
        </p>
      </div>

      <div className="portal-scroll-area flex-1 min-h-0">
        <div
          className={`dropzone ${dragActive ? "drag-active" : ""}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadCloud size={30} className="dropzone-icon text-teal-600 mb-1.5" />
          <div className="dropzone-text text-xs font-semibold text-slate-700">
            Select or drag & drop coral images or a .zip archive
          </div>
          <div className="dropzone-hint text-[11px] text-slate-400 mt-0.5">
            Supported formats: .jpg, .jpeg, .png, .webp, .zip
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.webp,.zip"
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                onFilesSelected(e.target.files);
              }
            }}
          />
        </div>

        {stagedImages.length > 0 && (
          <div className="mt-2.5">
            <div className="flex items-center justify-between mb-1.5">
              <div className="text-xs font-semibold text-emerald-700 flex items-center gap-1.5">
                <CheckCircle2 size={13} />
                <span>{stagedImages.length} image(s) ready for analysis</span>
              </div>
              <button
                type="button"
                className="text-[11px] font-medium text-slate-400 hover:text-red-600 transition-colors"
                onClick={onClearAll}
              >
                Clear all
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 my-1">
              {stagedImages.map((img, idx) => (
                <div
                  key={idx}
                  className="group relative rounded-xl border border-slate-200/90 bg-white overflow-hidden shadow-xs hover:shadow-md hover:border-teal-500/60 transition-all duration-200 flex flex-col"
                >
                  <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-100">
                    <img
                      src={img.dataUrl}
                      alt={img.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 ease-out"
                    />
                    <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded-md bg-slate-900/70 backdrop-blur-md text-white font-mono text-[10px] font-semibold tracking-wider shadow-xs">
                      #{String(idx + 1).padStart(2, "0")}
                    </div>
                    <button
                      type="button"
                      className="absolute top-2 right-2 h-5 w-5 rounded-full bg-slate-900/70 hover:bg-red-600 text-white flex items-center justify-center transition-colors shadow-xs"
                      onClick={() => onRemoveImage(idx)}
                      title="Remove image"
                    >
                      <X size={11} strokeWidth={2.5} />
                    </button>
                  </div>
                  <div className="px-2 py-1.5 bg-white flex items-center justify-between gap-1 border-t border-slate-100">
                    <span
                      className="text-[11px] font-medium text-slate-700 truncate"
                      title={img.name}
                    >
                      {img.name}
                    </span>
                    <span className="text-[9px] font-mono uppercase text-slate-400 bg-slate-50 border border-slate-200/60 px-1 py-0.2 rounded shrink-0">
                      {img.name.split(".").pop() || "IMG"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {stagedImages.length > 0 && (
        <BatchProgressCard
          itemsCount={stagedImages.length}
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
      )}
    </div>
  );
};
