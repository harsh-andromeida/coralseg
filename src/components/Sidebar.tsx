import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Sliders,
  Palette,
  Cpu,
  Minus,
  Plus,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { InfoTooltip } from "./InfoTooltip";
import { DeviceInfo, CoralSegment } from "../types";

interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onSetCollapsed?: (val: boolean) => void;
  pointsPerSide: number;
  onPointsPerSideChange: (val: number) => void;
  iouThresh: number;
  onIouThreshChange: (val: number) => void;
  stabilityThresh: number;
  onStabilityThreshChange: (val: number) => void;
  minAreaPx: number;
  onMinAreaPxChange: (val: number) => void;
  layoutMode: string;
  onLayoutModeChange: (val: string) => void;
  colorMode: string;
  onColorModeChange: (val: string) => void;
  alpha: number;
  onAlphaChange: (val: number) => void;
  drawContours: boolean;
  onDrawContoursChange: (val: boolean) => void;
  drawLabels: boolean;
  onDrawLabelsChange: (val: boolean) => void;
  drawBoxes: boolean;
  onDrawBoxesChange: (val: boolean) => void;
  segments: CoralSegment[];
  selectedMaskId: number | null;
  onSelectMaskId: (id: number | null) => void;
  deviceInfo: DeviceInfo | null;
  devicePreference: string;
  onDevicePreferenceChange: (val: string) => void;
}

const DEFAULT_SIDEBAR_WIDTH = 320;
const MIN_SIDEBAR_WIDTH = 240;
const COLLAPSE_THRESHOLD = 180;
const MAX_SIDEBAR_WIDTH = 540;
const STORAGE_KEY = "reefvision_sidebar_width";

const LAYOUT_OPTIONS = [
  { id: "Side-by-Side", label: "Side by Side" },
  { id: "Overlay Only", label: "Overlay" },
  { id: "Original Only", label: "Original" },
  { id: "Masks on Black", label: "Masks on Black" },
];

const COLOR_MODES = [
  { id: "instances", label: "Colony Instances", dot: "bg-[#0d7c85]" },
  { id: "bleaching", label: "Health Condition", dot: "bg-emerald-500" },
  { id: "taxonomy", label: "Taxonomy", dot: "bg-sky-500" },
  { id: "taxonomy_condition", label: "Taxonomy + Health", dot: "bg-indigo-500" },
];

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  onToggleCollapse,
  onSetCollapsed,
  pointsPerSide,
  onPointsPerSideChange,
  iouThresh,
  onIouThreshChange,
  stabilityThresh,
  onStabilityThreshChange,
  minAreaPx,
  onMinAreaPxChange,
  layoutMode,
  onLayoutModeChange,
  colorMode,
  onColorModeChange,
  alpha,
  onAlphaChange,
  drawContours,
  onDrawContoursChange,
  drawLabels,
  onDrawLabelsChange,
  drawBoxes,
  onDrawBoxesChange,
  segments,
  selectedMaskId,
  onSelectMaskId,
  deviceInfo,
  devicePreference,
  onDevicePreferenceChange,
}) => {
  const [width, setWidth] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = parseInt(saved, 10);
        if (!isNaN(parsed) && parsed >= MIN_SIDEBAR_WIDTH && parsed <= MAX_SIDEBAR_WIDTH) {
          return parsed;
        }
      }
    } catch {
      // Fallback
    }
    return DEFAULT_SIDEBAR_WIDTH;
  });

  const [isResizing, setIsResizing] = useState<boolean>(false);
  const widthRef = useRef(width);
  widthRef.current = width;

  const lastGoodWidthRef = useRef<number>(
    width >= MIN_SIDEBAR_WIDTH ? width : DEFAULT_SIDEBAR_WIDTH
  );
  if (width >= MIN_SIDEBAR_WIDTH) {
    lastGoodWidthRef.current = width;
  }

  const startResizing = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    setIsResizing(true);
    const startX = e.clientX;
    const startWidth = widthRef.current >= MIN_SIDEBAR_WIDTH ? widthRef.current : lastGoodWidthRef.current;

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const onPointerMove = (moveEvent: PointerEvent) => {
      const delta = moveEvent.clientX - startX;
      const tentative = startWidth + delta;
      const maxAllowed = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.45));

      if (tentative < COLLAPSE_THRESHOLD) {
        setWidth(0);
      } else {
        const clamped = Math.min(Math.max(tentative, MIN_SIDEBAR_WIDTH), maxAllowed);
        setWidth(clamped);
        lastGoodWidthRef.current = clamped;
      }
    };

    const onPointerUp = (upEvent: PointerEvent) => {
      setIsResizing(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);

      const delta = upEvent.clientX - startX;
      const tentative = startWidth + delta;

      if (tentative < COLLAPSE_THRESHOLD) {
        if (onSetCollapsed) {
          onSetCollapsed(true);
        } else {
          onToggleCollapse();
        }
        setWidth(lastGoodWidthRef.current);
      } else {
        const maxAllowed = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.45));
        const finalWidth = Math.min(Math.max(tentative, MIN_SIDEBAR_WIDTH), maxAllowed);
        setWidth(finalWidth);
        lastGoodWidthRef.current = finalWidth;
        try {
          localStorage.setItem(STORAGE_KEY, String(finalWidth));
        } catch {
          // ignore
        }
      }
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }, [onSetCollapsed, onToggleCollapse]);

  const startUncollapseDrag = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    setIsResizing(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const onPointerMove = (moveEvent: PointerEvent) => {
      if (moveEvent.clientX >= COLLAPSE_THRESHOLD) {
        if (onSetCollapsed) onSetCollapsed(false);
        else if (isCollapsed) onToggleCollapse();

        const maxAllowed = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.45));
        const next = Math.min(Math.max(moveEvent.clientX, MIN_SIDEBAR_WIDTH), maxAllowed);
        setWidth(next);
        lastGoodWidthRef.current = next;
      }
    };

    const onPointerUp = (upEvent: PointerEvent) => {
      setIsResizing(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);

      if (upEvent.clientX >= COLLAPSE_THRESHOLD) {
        if (onSetCollapsed) onSetCollapsed(false);
        else if (isCollapsed) onToggleCollapse();

        const maxAllowed = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.45));
        const finalWidth = Math.min(Math.max(upEvent.clientX, MIN_SIDEBAR_WIDTH), maxAllowed);
        setWidth(finalWidth);
        lastGoodWidthRef.current = finalWidth;
        try {
          localStorage.setItem(STORAGE_KEY, String(finalWidth));
        } catch {
          // ignore
        }
      }
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }, [isCollapsed, onSetCollapsed, onToggleCollapse]);

  const handleResetWidth = useCallback(() => {
    setWidth(DEFAULT_SIDEBAR_WIDTH);
    lastGoodWidthRef.current = DEFAULT_SIDEBAR_WIDTH;
    try {
      localStorage.setItem(STORAGE_KEY, String(DEFAULT_SIDEBAR_WIDTH));
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, []);

  if (isCollapsed) {
    return (
      <>
        <Button
          variant="outline"
          size="icon"
          onClick={() => {
            if (onSetCollapsed) onSetCollapsed(false);
            else onToggleCollapse();
            setWidth(lastGoodWidthRef.current);
          }}
          title="Open Analysis Controls"
          className="fixed top-16 left-4 z-40 h-9 w-9 rounded-lg bg-card shadow-md border-border text-foreground hover:text-primary transition-transform hover:scale-105"
        >
          <PanelLeftOpen className="h-4 w-4" />
        </Button>
        {/* Left edge drag strip to pull-open */}
        <div
          role="separator"
          aria-orientation="vertical"
          onPointerDown={startUncollapseDrag}
          title="Drag to open sidebar"
          className="absolute top-0 left-0 w-3 h-full cursor-col-resize z-40 select-none group flex items-center justify-center transition-colors"
        >
          <div className="w-[3px] h-14 rounded-r-full bg-border/80 group-hover:bg-primary group-hover:h-24 transition-all duration-150" />
        </div>
      </>
    );
  }

  const inferenceSliders = [
    {
      label: "Grid Density",
      value: pointsPerSide,
      min: 8,
      max: 36,
      step: 4,
      onChange: onPointsPerSideChange,
      format: (v: number) => `${v}`,
      tip: "Density of input prompt points sampled across the image grid.",
    },
    {
      label: "IoU Threshold",
      value: iouThresh,
      min: 0.2,
      max: 0.98,
      step: 0.02,
      onChange: onIouThreshChange,
      format: (v: number) => v.toFixed(2),
      tip: "Cutoff for predicted Intersection-over-Union quality.",
    },
    {
      label: "Stability Threshold",
      value: stabilityThresh,
      min: 0.2,
      max: 0.99,
      step: 0.01,
      onChange: onStabilityThreshChange,
      format: (v: number) => v.toFixed(2),
      tip: "Filters masks with unstable boundaries across binarization thresholds.",
    },
  ];

  return (
    <aside
      style={{
        width: `${width}px`,
        maxWidth: `${MAX_SIDEBAR_WIDTH}px`,
        minWidth: isResizing && width === 0 ? 0 : `${MIN_SIDEBAR_WIDTH}px`,
      }}
      className={cn(
        "relative shrink-0 h-full border-r border-border bg-card/60 flex flex-col select-none group/sidebar",
        isResizing ? "transition-none select-none" : "transition-[width] duration-75 ease-out",
        isResizing && width === 0 && "opacity-20 pointer-events-none"
      )}
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-border px-5 shrink-0 bg-card">
        <div className="flex items-center gap-2.5 font-semibold text-xs uppercase tracking-wider text-foreground">
          <Sliders className="h-4 w-4 text-primary" />
          <span>Analysis Controls</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleCollapse}
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          title="Collapse sidebar"
        >
          <PanelLeftClose className="h-4 w-4" />
        </Button>
      </div>

      {/* Accordion Controls with generous padding and clear hierarchy */}
      <div className="flex-1 overflow-y-auto px-5 py-3 text-xs">
        <Accordion
          type="multiple"
          defaultValue={["params", "display", "hardware"]}
          className="w-full flex flex-col gap-1"
        >
          {/* 1. Inference Parameters */}
          <AccordionItem value="params" className="border-b border-border/80">
            <AccordionTrigger className="py-3.5 text-xs font-semibold text-foreground hover:text-primary transition-colors">
              <div className="flex items-center gap-2">
                <Sliders className="h-3.5 w-3.5 text-primary" />
                <span>Inference Parameters</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="flex flex-col gap-4.5 pt-1 pb-5">
              {inferenceSliders.map((s) => (
                <div key={s.label} className="flex flex-col gap-2">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5 font-medium text-foreground">
                      <span>{s.label}</span>
                      <InfoTooltip content={s.tip} />
                    </div>
                    <span className="font-mono text-xs font-semibold text-primary bg-primary/10 px-2 py-0.5 rounded-md">
                      {s.format(s.value)}
                    </span>
                  </div>
                  <Slider
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={[s.value]}
                    onValueChange={(vals) => s.onChange(vals[0])}
                    className="py-1"
                  />
                </div>
              ))}

              {/* Min Mask Area */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5 font-medium text-foreground">
                    <span>Min Mask Area</span>
                    <InfoTooltip content="Minimum pixel count threshold for detected corals." />
                  </div>
                  <span className="font-mono text-xs font-semibold text-muted-foreground">
                    {minAreaPx.toLocaleString("en-US")} px
                  </span>
                </div>
                <div className="flex items-center rounded-lg border border-input bg-background shadow-xs overflow-hidden">
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-9 shrink-0 text-muted-foreground hover:text-foreground rounded-none"
                    onClick={() => onMinAreaPxChange(Math.max(10, minAreaPx - 50))}
                  >
                    <Minus className="h-3.5 w-3.5" />
                  </Button>
                  <Input
                    type="number"
                    min={10}
                    max={50000}
                    step={50}
                    value={minAreaPx}
                    onChange={(e) => {
                      const v = parseInt(e.target.value, 10);
                      if (!isNaN(v)) onMinAreaPxChange(Math.max(10, Math.min(50000, v)));
                    }}
                    className="h-8 border-0 shadow-none text-center font-mono text-xs focus-visible:ring-0 rounded-none bg-transparent"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-9 shrink-0 text-muted-foreground hover:text-foreground rounded-none"
                    onClick={() => onMinAreaPxChange(Math.min(50000, minAreaPx + 50))}
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* 2. Display Controls */}
          <AccordionItem value="display" className="border-b border-border/80">
            <AccordionTrigger className="py-3.5 text-xs font-semibold text-foreground hover:text-primary transition-colors">
              <div className="flex items-center gap-2">
                <Palette className="h-3.5 w-3.5 text-primary" />
                <span>Display Controls</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="flex flex-col gap-4.5 pt-1 pb-5">
              {/* Layout Mode */}
              <div className="flex flex-col gap-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  View Layout
                </div>
                <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-muted/60 border border-border/60">
                  {LAYOUT_OPTIONS.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => onLayoutModeChange(opt.id)}
                      className={`h-8 px-2 rounded-md text-xs font-medium transition-all ${
                        layoutMode === opt.id
                          ? "bg-card text-foreground shadow-xs font-semibold"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Color Mode */}
              <div className="flex flex-col gap-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Color Mode
                </div>
                <RadioGroup
                  value={colorMode}
                  onValueChange={onColorModeChange}
                  className="flex flex-col gap-2"
                >
                  {COLOR_MODES.map((m) => (
                    <div
                      key={m.id}
                      className={`flex items-center space-x-2.5 px-2.5 py-1.5 rounded-lg border transition-colors cursor-pointer ${
                        colorMode === m.id
                          ? "bg-primary/5 border-primary/30"
                          : "border-transparent hover:bg-muted/50"
                      }`}
                      onClick={() => onColorModeChange(m.id)}
                    >
                      <RadioGroupItem value={m.id} id={`color-${m.id}`} />
                      <Label
                        htmlFor={`color-${m.id}`}
                        className="text-xs font-medium text-foreground cursor-pointer flex items-center gap-2"
                      >
                        <span className={`h-2.5 w-2.5 rounded-full ${m.dot}`} />
                        <span>{m.label}</span>
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>

              {/* Overlay Opacity */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs font-medium text-foreground">
                  <span>Overlay Opacity</span>
                  <span className="font-mono text-xs font-semibold text-primary bg-primary/10 px-2 py-0.5 rounded-md">
                    {alpha.toFixed(2)}
                  </span>
                </div>
                <Slider
                  min={0.05}
                  max={1.0}
                  step={0.05}
                  value={[alpha]}
                  onValueChange={(vals) => onAlphaChange(vals[0])}
                  className="py-1"
                />
              </div>

              {/* Annotations */}
              <div className="flex flex-col gap-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Annotations
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="ann-borders"
                      checked={drawContours}
                      onCheckedChange={(c) => onDrawContoursChange(!!c)}
                    />
                    <Label htmlFor="ann-borders" className="text-xs font-medium text-foreground cursor-pointer">
                      Borders
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="ann-labels"
                      checked={drawLabels}
                      onCheckedChange={(c) => onDrawLabelsChange(!!c)}
                    />
                    <Label htmlFor="ann-labels" className="text-xs font-medium text-foreground cursor-pointer">
                      ID Badges
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2 col-span-2">
                    <Checkbox
                      id="ann-boxes"
                      checked={drawBoxes}
                      onCheckedChange={(c) => onDrawBoxesChange(!!c)}
                    />
                    <Label htmlFor="ann-boxes" className="text-xs font-medium text-foreground cursor-pointer">
                      Bounding Boxes
                    </Label>
                  </div>
                </div>
              </div>

              {/* Highlight Coral */}
              <div className="flex flex-col gap-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Filter / Highlight Coral
                </div>
                <Select
                  value={selectedMaskId !== null ? String(selectedMaskId) : "all"}
                  onValueChange={(val) => onSelectMaskId(val === "all" ? null : parseInt(val, 10))}
                >
                  <SelectTrigger className="h-9 text-xs bg-background">
                    <SelectValue placeholder="All Corals" />
                  </SelectTrigger>
                  <SelectContent position="popper" sideOffset={4}>
                    <SelectItem value="all">All Corals</SelectItem>
                    {segments.map((s) => (
                      <SelectItem key={s.id} value={String(s.id)}>
                        Coral #{s.id} [{s.genus}] ({s.condition})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* 3. Compute Engine */}
          <AccordionItem value="hardware" className="border-b-0">
            <AccordionTrigger className="py-3.5 text-xs font-semibold text-foreground hover:text-primary transition-colors">
              <div className="flex items-center gap-2">
                <Cpu className="h-3.5 w-3.5 text-primary" />
                <span>Compute Engine</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="flex flex-col gap-3 pt-1 pb-5">
              <div className="flex flex-col gap-2">
                <Label className="text-xs text-muted-foreground font-medium">Device Preference</Label>
                <Select value={devicePreference} onValueChange={onDevicePreferenceChange}>
                  <SelectTrigger className="h-9 text-xs bg-background">
                    <SelectValue placeholder="Select device" />
                  </SelectTrigger>
                  <SelectContent position="popper" sideOffset={4}>
                    <SelectItem value="auto">Auto (Recommended)</SelectItem>
                    <SelectItem value="gpu" disabled={!deviceInfo?.gpu_available}>
                      {deviceInfo?.gpu_available ? `GPU (${deviceInfo.gpu_name})` : "GPU [Unavailable]"}
                    </SelectItem>
                    <SelectItem value="multithread_cpu">
                      Multi-Thread CPU ({deviceInfo?.cpu_count || 4} Threads)
                    </SelectItem>
                    <SelectItem value="cpu">Single-Thread CPU</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      {/* Resizable drag handle on right border */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-valuenow={width}
        aria-valuemin={MIN_SIDEBAR_WIDTH}
        aria-valuemax={MAX_SIDEBAR_WIDTH}
        aria-label="Resize Analysis Controls sidebar"
        tabIndex={0}
        onPointerDown={startResizing}
        onDoubleClick={handleResetWidth}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") {
            const next = Math.max(width - 16, MIN_SIDEBAR_WIDTH);
            setWidth(next);
            lastGoodWidthRef.current = next;
            try {
              localStorage.setItem(STORAGE_KEY, String(next));
            } catch {}
          } else if (e.key === "ArrowRight") {
            const maxAllowed = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.45));
            const next = Math.min(width + 16, maxAllowed);
            setWidth(next);
            lastGoodWidthRef.current = next;
            try {
              localStorage.setItem(STORAGE_KEY, String(next));
            } catch {}
          }
        }}
        title="Drag to resize • Drag left to hide • Double-click to reset"
        className={cn(
          "absolute top-0 -right-2 w-4 h-full cursor-col-resize z-50 select-none flex items-center justify-center transition-colors group focus-visible:outline-none",
          isResizing && "bg-primary/20"
        )}
      >
        {/* Border line highlight on hover */}
        <div className="absolute top-0 right-[7px] w-[2px] h-full bg-transparent group-hover:bg-primary/60 transition-colors pointer-events-none" />
        {/* Centered pill indicator */}
        <div
          className={cn(
            "w-[3px] h-10 rounded-full bg-border/90 group-hover:bg-primary group-hover:h-24 group-hover:w-[3.5px] transition-all duration-150 shadow-xs z-10",
            isResizing && "bg-primary h-full w-[3px] rounded-none opacity-100"
          )}
        />
      </div>
    </aside>
  );
};
