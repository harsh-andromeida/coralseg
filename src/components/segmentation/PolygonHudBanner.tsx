import React from "react";
import { Button } from "@/components/ui/button";

interface PolygonHudBannerProps {
  onClosePolygon: () => void;
  onCancel: () => void;
}

export const PolygonHudBanner: React.FC<PolygonHudBannerProps> = ({
  onClosePolygon,
  onCancel,
}) => {
  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-white/95 text-slate-900 backdrop-blur-md px-4 py-2 rounded-xl border border-slate-200 shadow-xl text-xs">
      <span className="text-slate-700 font-medium">
        Click around coral boundary • Click start point or Close button to finish
      </span>
      <Button
        type="button"
        variant="default"
        size="sm"
        onClick={onClosePolygon}
        className="h-6 px-2.5 text-[11px] font-semibold bg-[#0d7c85] hover:bg-[#0b6b73] text-white"
      >
        Close Polygon
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onCancel}
        className="h-6 px-2 text-[11px] text-slate-500 hover:text-slate-800"
      >
        Cancel
      </Button>
    </div>
  );
};
