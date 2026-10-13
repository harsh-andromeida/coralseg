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
  const isGpu =
    deviceInfo?.device_type === "gpu" ||
    deviceInfo?.active_provider?.includes("CUDA");

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
            className="h-8 gap-2 px-3 py-1 font-mono text-xs font-medium text-foreground bg-muted/50 border-border/80"
            title={`Active Provider: ${deviceInfo.active_provider}`}
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            {isGpu ? (
              <Zap className="h-3.5 w-3.5 text-amber-500" />
            ) : (
              <Cpu className="h-3.5 w-3.5 text-primary" />
            )}
            <span>
              {deviceInfo.active_provider.includes("CUDA")
                ? "NVIDIA CUDA GPU"
                : deviceInfo.active_provider.includes("CoreML")
                ? "Apple CoreML"
                : "Multi-Threaded CPU Engine"}
            </span>
          </Badge>
        )}
      </div>
    </header>
  );
};
