import React from "react";
import { CheckCircle2, AlertCircle, Settings2, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DeviceInfo } from "../../types";
import { getEngineName } from "./HardwareAccelerationCard";

export interface SystemStatusSummaryProps {
  isBackendConnected: boolean;
  isCheckingBackend: boolean;
  allDownloaded: boolean;
  deviceInfo: DeviceInfo | null;
  onOpenSettings: () => void;
}

/** One-glance system status for the upload page; all configuration is in the Settings dialog. */
export const SystemStatusSummary: React.FC<SystemStatusSummaryProps> = ({
  isBackendConnected,
  isCheckingBackend,
  allDownloaded,
  deviceInfo,
  onOpenSettings,
}) => {
  const ready = isBackendConnected && allDownloaded;
  const headline = isCheckingBackend
    ? "Connecting to vision engine..."
    : !isBackendConnected
    ? "Vision engine offline"
    : allDownloaded
    ? "Ready"
    : "Models need to be downloaded";

  return (
    <div className="mt-3 rounded-xl border border-slate-200/90 bg-slate-50/50 p-3 shadow-xs space-y-2.5">
      <div className="flex items-center gap-2 text-xs font-semibold text-slate-800">
        {ready ? (
          <CheckCircle2 size={14} className="text-emerald-600" />
        ) : (
          <AlertCircle size={14} className={isBackendConnected || isCheckingBackend ? "text-amber-500" : "text-rose-500"} />
        )}
        <span>{headline}</span>
      </div>
      <div className="flex items-center gap-2 text-[11px] text-slate-600">
        <Cpu size={12} className="text-cyan-600 shrink-0" />
        <span className="truncate">{getEngineName(deviceInfo)}</span>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={onOpenSettings}
        className="h-7 w-full text-xs font-semibold gap-1.5"
      >
        <Settings2 size={12} />
        <span>Settings</span>
      </Button>
    </div>
  );
};
