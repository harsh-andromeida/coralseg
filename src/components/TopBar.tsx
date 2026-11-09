import React from "react";
import { ArrowLeft, Cpu, Zap } from "lucide-react";
import { BrandLogo } from "./BrandLogo";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { DeviceInfo } from "../types";

interface TopBarProps {
  imageName?: string;
  currentIndex?: number;
  totalImages?: number;
  onBack: () => void;
  isSidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
  deviceInfo?: DeviceInfo | null;
}

export const TopBar: React.FC<TopBarProps> = ({ onBack, deviceInfo }) => {
  const isCudaActive = Boolean(deviceInfo?.active_provider?.includes("CUDA"));
  const isCudaPendingOrt = Boolean(
    (deviceInfo?.torch_cuda_available || deviceInfo?.cuda_status === "onnxruntime_gpu_missing") && !isCudaActive
  );
  const isCoreML = Boolean(deviceInfo?.active_provider?.includes("CoreML") || deviceInfo?.gpu_name?.includes("MPS"));

  let badgeLabel = "Multi-Threaded CPU Engine";
  let badgeTitle = `Active Provider: ${deviceInfo?.active_provider || "CPU"}`;
  if (isCudaActive) {
    badgeLabel = "NVIDIA CUDA GPU";
    badgeTitle = "Hardware Accelerated: NVIDIA CUDA Tensor Cores Active";
  } else if (isCudaPendingOrt) {
    badgeLabel = "CPU (NVIDIA GPU Needs onnxruntime-gpu)";
    badgeTitle = "NVIDIA GPU detected via PyTorch, but onnxruntime-gpu is required for ONNX models.";
  } else if (isCoreML) {
    badgeLabel = "Apple Silicon (MPS)";
    badgeTitle = "Hardware Accelerated: Apple Silicon Neural Engine & GPU";
  }

  return (
    <header className="sticky top-0 z-30 flex h-14 w-full shrink-0 items-center justify-between border-b border-border bg-card/80 backdrop-blur-md px-6 shadow-xs">
      {/* Left: Back to Selection */}
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          title="Return to image library"
          className="h-8 gap-2 px-2.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/80 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Library</span>
        </Button>
      </div>

      {/* Center: Brand Identity */}
      <div className="flex items-center justify-center">
        <BrandLogo variant="topbar" />
      </div>

      {/* Right: Hardware Acceleration Badge */}
      <div className="flex items-center gap-2 shrink-0">
        {deviceInfo && (
          <Badge
            variant="outline"
            className={`h-8 gap-2 px-3 py-1 font-mono text-xs font-medium text-foreground bg-muted/50 border-border/80 ${
              isCudaPendingOrt ? "border-amber-300 bg-amber-50/50" : ""
            }`}
            title={badgeTitle}
          >
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full ${
                  isCudaPendingOrt ? "bg-amber-400" : "bg-emerald-400"
                } opacity-75`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  isCudaPendingOrt ? "bg-amber-500" : "bg-emerald-500"
                }`}
              />
            </span>
            {isCudaActive ? (
              <Zap className="h-3.5 w-3.5 text-amber-500" />
            ) : isCudaPendingOrt ? (
              <Zap className="h-3.5 w-3.5 text-amber-500 animate-pulse" />
            ) : isCoreML ? (
              <Zap className="h-3.5 w-3.5 text-emerald-500" />
            ) : (
              <Cpu className="h-3.5 w-3.5 text-primary" />
            )}
            <span className="truncate max-w-[260px]">{badgeLabel}</span>
          </Badge>
        )}
      </div>
    </header>
  );
};
