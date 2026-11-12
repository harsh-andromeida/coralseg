import React from "react";
import {
  Check,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Zap,
  Trash2,
} from "lucide-react";
import { UISelect } from "../UISelect";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { DeviceInfo, ModelSpec, DownloadProgress } from "../../types";
import { getCustomApiUrl } from "../../services/api";
import { MODEL_GROUPS, isGroupReady } from "./modelGroups";

export const getEngineName = (info: DeviceInfo | null): string => {
  if (!info) return "Detecting compute hardware...";
  if (info.active_provider?.includes("CUDA")) {
    return "NVIDIA CUDA GPU";
  }
  if (info.cuda_status === "onnxruntime_gpu_missing") {
    return "CPU (GPU Needs onnxruntime-gpu)";
  }
  if (info.active_provider?.includes("CoreML") || info.gpu_name?.includes("MPS")) {
    return "Apple Silicon (MPS)";
  }
  if (info.mode === "cpu") {
    return "Single-Threaded CPU Engine (1 Thread)";
  }
  if (info.cpu_count && info.cpu_count > 1) {
    return `Multi-Threaded CPU Engine (${info.cpu_count} Threads)`;
  }
  return info.gpu_name || "CPU Engine";
};

export interface HardwareAccelerationCardProps {
  isBackendConnected: boolean;
  isCheckingBackend: boolean;
  allDownloaded: boolean;
  modelsList: ModelSpec[];
  deviceInfo: DeviceInfo | null;
  devicePref: string;
  concurrency: number;
  downloadProgress: DownloadProgress;
  isDeletingModel: string | null;
  onManualRetry: () => void;
  onOpenRemoteModal: () => void;
  onOpenDeleteModal: () => void;
  onDeleteModel: (target: string) => void;
  onDeviceChange: (val: string) => void;
  onConcurrencyChange: (val: number) => void;
}

export const HardwareAccelerationCard: React.FC<HardwareAccelerationCardProps> = ({
  isBackendConnected,
  isCheckingBackend,
  allDownloaded,
  modelsList,
  deviceInfo,
  devicePref,
  concurrency,
  downloadProgress,
  isDeletingModel,
  onManualRetry,
  onOpenRemoteModal,
  onOpenDeleteModal,
  onDeleteModel,
  onDeviceChange,
  onConcurrencyChange,
}) => {
  return (
    <>
      {/* System & Model Status Panel */}
      <div className="rounded-xl border border-slate-200/90 bg-slate-50/50 p-2.5 shadow-xs mb-2.5">
        {/* Row 1: Vision Engine Connected */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            {isBackendConnected ? (
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
            ) : (
              <span
                className={`h-2 w-2 rounded-full ${
                  isCheckingBackend ? "bg-amber-400 animate-pulse" : "bg-red-500"
                }`}
              />
            )}
            <span className="text-xs font-bold text-slate-800">Vision Engine</span>
          </div>

          {isBackendConnected ? (
            <div className="flex items-center gap-1.5">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/70">
                <Check size={10} strokeWidth={2.5} />
                Connected
              </span>
              {getCustomApiUrl() && (
                <button
                  type="button"
                  onClick={onOpenRemoteModal}
                  className="text-[9.5px] font-medium text-teal-700 hover:underline px-1 py-0.5 bg-teal-50 rounded border border-teal-200/60"
                  title={`Connected to remote backend: ${getCustomApiUrl()}`}
                >
                  Remote
                </button>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-medium text-amber-700">
                {isCheckingBackend ? "Connecting..." : "Offline"}
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-6 px-1.5 text-[10px] font-semibold border-amber-300 hover:bg-amber-50"
                onClick={onManualRetry}
              >
                Retry
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-6 px-1.5 text-[10px] font-semibold text-teal-700 hover:bg-teal-50"
                onClick={onOpenRemoteModal}
                title="Connect to Google Colab or remote tunnel"
              >
                Remote/Colab
              </Button>
            </div>
          )}
        </div>

        {isBackendConnected && (
          <>
            <div className="h-px bg-slate-200/70 my-2" />

            {/* Row 2: Foundation Models header + Delete/Manage Button */}
            <div className="flex items-center justify-between gap-2 mb-1.5">
              <div className="flex items-center gap-1.5">
                {allDownloaded ? (
                  <CheckCircle2 size={13} className="text-emerald-600" />
                ) : (
                  <AlertCircle size={13} className="text-amber-600" />
                )}
                <span className="text-xs font-bold text-slate-800">Foundation Models</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span
                  className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-md border ${
                    allDownloaded
                      ? "text-emerald-700 bg-emerald-50 border-emerald-200/70"
                      : "text-amber-700 bg-amber-50 border-amber-200/70"
                  }`}
                >
                  {allDownloaded ? "Ready" : "Download Required"}
                </span>
                {allDownloaded && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-6 px-2 text-[10px] font-semibold text-slate-500 hover:text-red-600 hover:bg-red-50 hover:border-red-200 border-slate-200 gap-1 rounded-md transition-colors"
                    onClick={onOpenDeleteModal}
                    disabled={downloadProgress.is_downloading || !!isDeletingModel}
                    title="Manage and delete individual foundation models"
                  >
                    <Trash2 className="h-3 w-3" />
                    <span>Manage</span>
                  </Button>
                )}
              </div>
            </div>

            {/* Row 3: Models list */}
            <div className="flex flex-col gap-1">
              {MODEL_GROUPS.map((group) => {
                const ready = isGroupReady(group.id, allDownloaded, modelsList);
                return (
                  <div
                    key={group.id}
                    className="flex items-center justify-between px-2 py-1 rounded-lg bg-white border border-slate-200/70 text-xs hover:border-slate-300 transition-colors shadow-2xs"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      {ready ? (
                        <Check size={12} className="text-emerald-600 shrink-0" strokeWidth={2.5} />
                      ) : (
                        <AlertCircle size={12} className="text-amber-500 shrink-0" />
                      )}
                      <span
                        className="font-semibold text-slate-700 text-[11px] truncate"
                        title={group.fullName}
                      >
                        {group.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0 ml-2">
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-50 border border-slate-200/60 px-1.5 py-0.5 rounded">
                        {group.size}
                      </span>
                      {ready && (
                        <button
                          type="button"
                          className="text-slate-300 hover:text-red-500 transition-colors p-0.5 rounded"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteModel(group.id);
                          }}
                          disabled={downloadProgress.is_downloading || !!isDeletingModel}
                          title={`Delete ${group.name} (${group.size})`}
                        >
                          <Trash2
                            size={11}
                            className={isDeletingModel === group.id ? "animate-spin text-red-500" : ""}
                          />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {downloadProgress.is_downloading && (
              <div className="mt-2.5 p-2.5 rounded-lg bg-white border border-slate-200/80">
                <div className="flex justify-between items-center text-[11px] font-semibold text-slate-800 mb-1">
                  <span className="truncate max-w-[200px]">{downloadProgress.current_file}</span>
                  <span className="text-teal-700 font-mono">{downloadProgress.overall_pct}%</span>
                </div>
                <Progress
                  value={downloadProgress.overall_pct}
                  className="h-1.5 bg-slate-100 my-1"
                  indicatorColor="bg-[#0d7c85]"
                />
                <div className="text-[10px] text-slate-500">{downloadProgress.detail}</div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Hardware Acceleration Subcard */}
      <div className="rounded-xl border border-slate-200/90 bg-white p-2.5 shadow-xs">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <Cpu size={14} className="text-teal-600" />
            <span className="text-[11px] font-bold text-[#0f1e4a] tracking-wider uppercase">
              Hardware Acceleration
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <span className="text-[10px] font-semibold text-emerald-700 uppercase tracking-wide">
              Active
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between gap-2 p-1.5 rounded-lg bg-slate-50 border border-slate-100 mb-2">
          <span className="text-[11px] font-medium text-slate-500">Active Engine</span>
          <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-semibold bg-white border border-slate-200 text-slate-700 shadow-2xs">
            {deviceInfo?.active_provider?.includes("CUDA") ? (
              <Zap size={11} className="text-amber-500" />
            ) : deviceInfo?.cuda_status === "onnxruntime_gpu_missing" ? (
              <Zap size={11} className="text-amber-500 animate-pulse" />
            ) : (
              <Cpu size={11} className="text-teal-600" />
            )}
            <span className="truncate max-w-[170px]">{getEngineName(deviceInfo)}</span>
          </div>
        </div>

        {deviceInfo?.cuda_status === "onnxruntime_gpu_missing" && (
          <div className="p-2 mb-2 rounded-lg bg-amber-50/90 border border-amber-200 text-amber-900 text-[11px] flex flex-col gap-1.5 shadow-2xs">
            <div className="flex items-center gap-1.5 font-semibold text-amber-950">
              <Zap size={12} className="text-amber-600 fill-amber-500" />
              <span>NVIDIA GPU Detected (CPU Fallback)</span>
            </div>
            <p className="text-[10px] text-amber-800 leading-tight">
              PyTorch detects CUDA, but ONNX is running on CPU because{" "}
              <code className="bg-amber-100/80 px-1 py-0.5 rounded font-mono">
                onnxruntime-gpu
              </code>{" "}
              is not installed. To unlock 50x–100x GPU acceleration:
            </p>
            <div className="flex items-center justify-between gap-1 bg-white p-1.5 rounded border border-amber-200">
              <code className="text-[9px] font-mono text-slate-700 select-all truncate">
                pip uninstall -y onnxruntime && pip install onnxruntime-gpu
              </code>
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(
                    "pip uninstall -y onnxruntime && pip install onnxruntime-gpu"
                  );
                  alert("Command copied to clipboard! Run it in your Python / Colab environment.");
                }}
                className="px-2 py-0.5 text-[9px] font-semibold bg-amber-100 hover:bg-amber-200 text-amber-800 rounded transition-colors shrink-0"
              >
                Copy
              </button>
            </div>
          </div>
        )}

        <div className="mb-2">
          <label className="block text-[11px] font-medium text-slate-500 mb-1">
            Device Preference
          </label>
          <UISelect
            value={devicePref}
            onChange={onDeviceChange}
            options={[
              {
                value: "auto",
                label: deviceInfo?.gpu_available
                  ? `Auto (GPU: ${deviceInfo.gpu_name})`
                  : deviceInfo?.cpu_count && deviceInfo.cpu_count > 1
                  ? `Auto (Multi-Threaded CPU - ${deviceInfo.cpu_count} Threads)`
                  : "Auto (CPU)",
              },
              {
                value: "gpu",
                label: `GPU Acceleration ${
                  deviceInfo?.gpu_available ? `(${deviceInfo.gpu_name})` : "[Unavailable]"
                }`,
              },
              {
                value: "multithread_cpu",
                label: `Multi-Threaded CPU Engine (${deviceInfo?.cpu_count || 4} Threads)`,
              },
              {
                value: "cpu",
                label: "Single-Threaded CPU Engine (1 Thread)",
              },
            ]}
            placeholder="Select device"
          />
        </div>

        <div className="mb-2">
          <label className="block text-[11px] font-medium text-slate-500 mb-1">
            Batch Concurrency
          </label>
          <UISelect
            value={String(concurrency)}
            onChange={(val) => onConcurrencyChange(Number(val))}
            options={[
              {
                value: "1",
                label: "1 Worker (Sequential - Safe)",
              },
              {
                value: "2",
                label: "2 Workers (Dual Parallel)",
              },
              {
                value: "3",
                label: "3 Workers (Tri Parallel)",
              },
              {
                value: "4",
                label: "4 Workers (Quad Parallel - Fast)",
              },
            ]}
            placeholder="Select concurrency"
          />
        </div>

        <div className="text-[10px] text-slate-400 leading-tight pt-1 border-t border-slate-100">
          {deviceInfo?.gpu_available
            ? "Tensor operations accelerated natively via local GPU execution provider."
            : "Parallel tensor operations accelerated across all CPU cores via ONNX Runtime."}
        </div>
      </div>
    </>
  );
};
