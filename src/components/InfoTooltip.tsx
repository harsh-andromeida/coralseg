import React from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";

interface InfoTooltipProps {
  content: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({
  content,
  side = "top",
}) => {
  return (
    <TooltipProvider delayDuration={100}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className="inline-flex items-center justify-center text-slate-400 hover:text-[#0d7c85] transition-colors cursor-help p-0.5 rounded-full hover:bg-slate-100"
            aria-label="More information"
            onClick={(e) => e.preventDefault()}
          >
            <Info size={13} />
          </button>
        </TooltipTrigger>
        <TooltipContent side={side} sideOffset={5}>
          {content}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};
