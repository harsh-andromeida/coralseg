import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImage,
  Line,
  Circle,
  Group,
  Text as KonvaText,
  Rect,
} from "react-konva";
import Konva from "konva";
import {
  X,
  Save,
  Check,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Trash2,
  Plus,
  MousePointer,
  Wand2,
  Pentagon,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Sparkles,
  Sliders,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CoralSegment, PromptPoint, SummaryStats, HealthSummary } from "../types";
import {
  getSegmentPolygons,
  promptSegment,
  saveAnnotations,
} from "../services/api";

// Available Coral Taxa matching ReefVision taxonomy database
export const TAXONOMY_OPTIONS = [
  { genus: "Acropora", form: "Branching", common: "Staghorn / Table Coral" },
  { genus: "Porites", form: "Massive", common: "Massive / Lobe / Finger Coral" },
  { genus: "Montipora", form: "Foliose / Encrusting", common: "Plate / Encrusting Coral" },
  { genus: "Pocillopora", form: "Branching", common: "Cauliflower Coral" },
  { genus: "Stylophora", form: "Branching", common: "Cat's Paw Coral" },
  { genus: "Favia / Favites", form: "Massive", common: "Brain / Moon Coral" },
  { genus: "Platygyra", form: "Meandering", common: "Maze / Brain Coral" },
  { genus: "Turbinaria", form: "Foliose", common: "Scroll / Disc Coral" },
  { genus: "Millepora", form: "Branching / Encrusting", common: "Fire Coral" },
  { genus: "Pavona", form: "Foliose", common: "Cactus Coral" },
  { genus: "Fungia", form: "Solitary", common: "Mushroom Coral" },
  { genus: "Goniopora", form: "Massive", common: "Flowerpot Coral" },
  { genus: "Seriatopora", form: "Branching", common: "Bird's Nest Coral" },
  { genus: "Lobophyllia", form: "Flabello-meandroid", common: "Lobed Brain Coral" },
  { genus: "Galaxea", form: "Massive", common: "Star Coral" },
  { genus: "Diploastrea", form: "Massive", common: "Honeycomb Coral" },
  { genus: "Other Coral", form: "Other", common: "Unspecified Coral Genus" },
];

export const GROWTH_FORMS = [
  "Branching",
  "Massive",
  "Encrusting",
  "Foliose",
  "Tabular",
  "Columnar",
  "Meandering",
  "Solitary",
  "Flabello-meandroid",
  "Foliose / Encrusting",
  "Branching / Encrusting",
  "Other",
];

export const DISTINCT_COLORS = [
  "#00e5ff",
  "#ff007f",
  "#76ff03",
  "#ffd600",
  "#d500f9",
  "#ff6d00",
  "#00b0ff",
  "#00e676",
  "#ff1744",
  "#651fff",
  "#1de9b6",
  "#ff9100",
  "#f50057",
  "#00e5ff",
  "#c6ff00",
  "#2979ff",
];

interface SegmentationEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageName: string;
  imageSrc: string;
  imageResolution?: string;
  initialSegments?: CoralSegment[];
  onSaveSuccess?: (data: {
    summary: SummaryStats;
    segments: CoralSegment[];
    health_summary: HealthSummary;
    overlay_base64: string;
  }) => void;
  onNavigatePrev?: () => void;
  onNavigateNext?: () => void;
  hasPrev?: boolean;
  hasNext?: boolean;
}

type ToolMode = "select" | "sam" | "polygon";

interface EditableSegment {
  id: number;
  id_str: string;
  genus: string;
  growth_form: string;
  condition: "Healthy" | "Bleached";
  color_hex: string;
  polygon: [number, number][];
  area_px: number;
  area_pct: number;
  visible: boolean;
  is_human_annotated: boolean;
}

// Deep clone segments to ensure immutable history snapshots
const cloneSegments = (items: EditableSegment[]): EditableSegment[] => {
  return items.map((s) => ({
    ...s,
    polygon: s.polygon.map(([x, y]) => [x, y] as [number, number]),
  }));
};

export const SegmentationEditorModal: React.FC<SegmentationEditorModalProps> = ({
  isOpen,
  onClose,
  imageName,
  imageSrc,
  imageResolution = "0x0",
  initialSegments = [],
  onSaveSuccess,
  onNavigatePrev,
  onNavigateNext,
  hasPrev = false,
  hasNext = false,
}) => {
  // Canvas image element
  const [htmlImage, setHtmlImage] = useState<HTMLImageElement | null>(null);
  const [imgDimensions, setImgDimensions] = useState<{ w: number; h: number }>({
    w: 1920,
    h: 1080,
  });

  // Segments state
  const [segments, setSegments] = useState<EditableSegment[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // Active Tool Mode: 'select' (V), 'sam' (W), 'polygon' (P)
  const [toolMode, setToolMode] = useState<ToolMode>("select");

  // SAM Smart Click state
  const [samPoints, setSamPoints] = useState<PromptPoint[]>([]);
  const [samPointType, setSamPointType] = useState<1 | 0>(1); // 1 = Foreground (+), 0 = Background (-)
  const [isSamLoading, setIsSamLoading] = useState<boolean>(false);
  const [samPendingPolygon, setSamPendingPolygon] = useState<[number, number][] | null>(null);

  // Manual Polygon Draw state
  const [drawPoints, setDrawPoints] = useState<[number, number][]>([]);
  const [mouseCanvasPos, setMouseCanvasPos] = useState<[number, number] | null>(null);

  // Save / Dirty status
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<"saved" | "unsaved" | "saving">("saved");
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState<boolean>(false);
  const [lastSavedTime, setLastSavedTime] = useState<string | null>(null);

  // History stack for Undo / Redo (Ref-backed for zero stale closure bugs)
  const historyRef = useRef<EditableSegment[][]>([]);
  const historyIdxRef = useRef<number>(-1);
  const segmentsRef = useRef<EditableSegment[]>([]);
  const autoSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [historyIdx, setHistoryIdx] = useState<number>(-1);
  const [historyLength, setHistoryLength] = useState<number>(0);

  // Form inputs for active or new coral
  const [formGenus, setFormGenus] = useState<string>("Acropora");
  const [formGrowthForm, setFormGrowthForm] = useState<string>("Branching");
  const [formCondition, setFormCondition] = useState<"Healthy" | "Bleached">("Healthy");

  // Canvas Viewport transform (zoom / pan)
  const [stageScale, setStageScale] = useState<number>(1);
  const [stagePos, setStagePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [maskOpacity, setMaskOpacity] = useState<number>(0.45);

  const stageRef = useRef<Konva.Stage | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Keep segmentsRef in sync
  useEffect(() => {
    segmentsRef.current = segments;
  }, [segments]);

  // Load HTML image
  useEffect(() => {
    if (!isOpen || !imageSrc) return;
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = imageSrc;
    img.onload = () => {
      setHtmlImage(img);
      setImgDimensions({ w: img.naturalWidth, h: img.naturalHeight });
    };
  }, [isOpen, imageSrc]);

  // Calculate polygon area (Shoelace formula)
  const calculatePolygonArea = useCallback(
    (poly: [number, number][]): { area_px: number; area_pct: number } => {
      if (!poly || poly.length < 3) return { area_px: 0, area_pct: 0 };
      let sum = 0;
      for (let i = 0; i < poly.length; i++) {
        const j = (i + 1) % poly.length;
        sum += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
      }
      const area_px = Math.round(Math.abs(sum) / 2);
      const totalPixels = Math.max(imgDimensions.w * imgDimensions.h, 1);
      const area_pct = Number(((area_px / totalPixels) * 100).toFixed(2));
      return { area_px, area_pct };
    },
    [imgDimensions.w, imgDimensions.h]
  );

  // Push new state to history stack
  const pushHistory = useCallback((newSegments: EditableSegment[]) => {
    const cloned = cloneSegments(newSegments);
    const curIdx = historyIdxRef.current;
    const nextHistory = historyRef.current.slice(0, curIdx + 1);
    nextHistory.push(cloned);

    // Limit history stack size to 50
    if (nextHistory.length > 50) {
      nextHistory.shift();
    }

    historyRef.current = nextHistory;
    const nextIdx = nextHistory.length - 1;
    historyIdxRef.current = nextIdx;
    setHistoryIdx(nextIdx);
    setHistoryLength(nextHistory.length);
    setHasUnsavedChanges(true);
    setSaveStatus("unsaved");
  }, []);

  // Debounced Autosave (750ms after user stops making edits)
  const triggerDebouncedAutoSave = useCallback(
    (itemsToSave: EditableSegment[]) => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
      setHasUnsavedChanges(true);
      setSaveStatus("unsaved");

      autoSaveTimeoutRef.current = setTimeout(async () => {
        try {
          setSaveStatus("saving");
          const payload = itemsToSave.map((s) => ({
            id: s.id,
            genus: s.genus,
            growth_form: s.growth_form,
            condition: s.condition,
            color_hex: s.color_hex,
            polygon: s.polygon,
          }));
          const res = await saveAnnotations(imageName, payload);
          setHasUnsavedChanges(false);
          setSaveStatus("saved");
          setLastSavedTime(new Date().toLocaleTimeString());
          if (onSaveSuccess && res) {
            onSaveSuccess({
              summary: res.summary,
              segments: res.segments,
              health_summary: res.health_summary,
              overlay_base64: res.overlay_base64,
            });
          }
        } catch (e) {
          console.warn("Autosave warning:", e);
          setSaveStatus("unsaved");
        }
      }, 750);
    },
    [imageName, onSaveSuccess]
  );

  // Explicit Save Annotations Button (Ctrl + S / Cmd + S)
  const handleExplicitSave = useCallback(async () => {
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
    }
    setIsSaving(true);
    setSaveStatus("saving");
    try {
      const current = segmentsRef.current;
      const payload = current.map((s) => ({
        id: s.id,
        genus: s.genus,
        growth_form: s.growth_form,
        condition: s.condition,
        color_hex: s.color_hex,
        polygon: s.polygon,
      }));
      const res = await saveAnnotations(imageName, payload);
      setHasUnsavedChanges(false);
      setSaveStatus("saved");
      setLastSavedTime(new Date().toLocaleTimeString());
      if (onSaveSuccess && res) {
        onSaveSuccess({
          summary: res.summary,
          segments: res.segments,
          health_summary: res.health_summary,
          overlay_base64: res.overlay_base64,
        });
      }
    } catch (err: any) {
      setSaveStatus("unsaved");
      alert(`Save failed: ${err?.message || err}`);
    } finally {
      setIsSaving(false);
    }
  }, [imageName, onSaveSuccess]);

  // Undo (Ctrl+Z or Cmd+Z)
  const handleUndo = useCallback(() => {
    // If in SAM mode with un-run points, undo the last placed prompt point
    if (toolMode === "sam" && samPoints.length > 0 && !samPendingPolygon) {
      setSamPoints((prev) => prev.slice(0, -1));
      return;
    }
    if (historyIdxRef.current > 0) {
      const prevIdx = historyIdxRef.current - 1;
      historyIdxRef.current = prevIdx;
      setHistoryIdx(prevIdx);
      const restored = cloneSegments(historyRef.current[prevIdx]);
      setSegments(restored);
      segmentsRef.current = restored;
      triggerDebouncedAutoSave(restored);
    }
  }, [toolMode, samPoints.length, samPendingPolygon, triggerDebouncedAutoSave]);

  // Redo (Ctrl+Y, Cmd+Shift+Z, Ctrl+Shift+Z)
  const handleRedo = useCallback(() => {
    if (historyIdxRef.current < historyRef.current.length - 1) {
      const nextIdx = historyIdxRef.current + 1;
      historyIdxRef.current = nextIdx;
      setHistoryIdx(nextIdx);
      const restored = cloneSegments(historyRef.current[nextIdx]);
      setSegments(restored);
      segmentsRef.current = restored;
      triggerDebouncedAutoSave(restored);
    }
  }, [triggerDebouncedAutoSave]);

  // Fit image to canvas container
  const handleFitToScreen = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth === 0 || clientHeight === 0) return;

    const scaleX = (clientWidth - 48) / imgDimensions.w;
    const scaleY = (clientHeight - 48) / imgDimensions.h;
    const fitScale = Math.min(scaleX, scaleY, 1.0);

    const posX = (clientWidth - imgDimensions.w * fitScale) / 2;
    const posY = (clientHeight - imgDimensions.h * fitScale) / 2;

    setStageScale(Number(fitScale.toFixed(3)));
    setStagePos({ x: Math.round(posX), y: Math.round(posY) });
  }, [imgDimensions]);

  // Load polygons from backend or fallback to initialSegments
  const loadPolygons = useCallback(async () => {
    try {
      const res = await getSegmentPolygons(imageName);
      if (res && res.segments && res.segments.length > 0) {
        const loaded: EditableSegment[] = res.segments.map((s, idx) => ({
          id: s.id,
          id_str: s.id_str || `#${s.id}`,
          genus: s.genus || "Coral",
          growth_form: s.growth_form || "-",
          condition: s.condition === "Bleached" ? "Bleached" : "Healthy",
          color_hex: s.color_hex || DISTINCT_COLORS[idx % DISTINCT_COLORS.length],
          polygon: s.polygon && s.polygon.length >= 3 ? (s.polygon as [number, number][]) : [],
          area_px: s.area_px || 0,
          area_pct: s.area_pct || 0,
          visible: true,
          is_human_annotated: s.is_human_annotated || false,
        }));
        const cloned = cloneSegments(loaded);
        setSegments(cloned);
        segmentsRef.current = cloned;
        historyRef.current = [cloneSegments(cloned)];
        historyIdxRef.current = 0;
        setHistoryIdx(0);
        setHistoryLength(1);
        setHasUnsavedChanges(false);
        setSaveStatus("saved");
        if (cloned.length > 0) {
          setSelectedId(cloned[0].id);
        }
        return;
      }
    } catch {
      // Fallback to initialSegments if available
    }

    if (initialSegments.length > 0) {
      const fallback: EditableSegment[] = initialSegments.map((s, idx) => ({
        id: s.id,
        id_str: s.id_str || `#${s.id}`,
        genus: s.genus || "Coral",
        growth_form: s.growth_form || "-",
        condition: s.condition === "Bleached" ? "Bleached" : "Healthy",
        color_hex: s.color_hex || DISTINCT_COLORS[idx % DISTINCT_COLORS.length],
        polygon: s.polygon && s.polygon.length >= 3 ? (s.polygon as [number, number][]) : [],
        area_px: s.area_px || 0,
        area_pct: s.area_pct || 0,
        visible: true,
        is_human_annotated: s.is_human_annotated || false,
      }));
      const cloned = cloneSegments(fallback);
      setSegments(cloned);
      segmentsRef.current = cloned;
      historyRef.current = [cloneSegments(cloned)];
      historyIdxRef.current = 0;
      setHistoryIdx(0);
      setHistoryLength(1);
      setHasUnsavedChanges(false);
      setSaveStatus("saved");
      if (cloned.length > 0) {
        setSelectedId(cloned[0].id);
      }
    } else {
      setSegments([]);
      segmentsRef.current = [];
      historyRef.current = [[]];
      historyIdxRef.current = 0;
      setHistoryIdx(0);
      setHistoryLength(1);
      setHasUnsavedChanges(false);
      setSaveStatus("saved");
      setSelectedId(null);
    }
  }, [imageName, initialSegments]);

  useEffect(() => {
    if (isOpen) {
      loadPolygons();
      setToolMode("select");
      setSamPoints([]);
      setSamPendingPolygon(null);
      setDrawPoints([]);
      handleFitToScreen();
    }
  }, [isOpen, imageName, loadPolygons, handleFitToScreen]);

  // Synchronize form inputs when selected coral changes
  useEffect(() => {
    if (selectedId !== null) {
      const target = segments.find((s) => s.id === selectedId);
      if (target) {
        setFormGenus(target.genus);
        setFormGrowthForm(target.growth_form);
        setFormCondition(target.condition);
      }
    }
  }, [selectedId, segments]);

  // Mouse wheel zoom anchored to cursor
  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };

    const factor = e.evt.deltaY < 0 ? 1.15 : 0.88;
    const newScale = Math.min(Math.max(Number((oldScale * factor).toFixed(3)), 0.2), 8.0);

    setStageScale(newScale);
    setStagePos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  // Convert Stage pointer coordinates to natural image space
  const getStageRelativePos = useCallback((): [number, number] | null => {
    const stage = stageRef.current;
    if (!stage) return null;
    const pointer = stage.getPointerPosition();
    if (!pointer) return null;
    const transform = stage.getAbsoluteTransform().copy().invert();
    const pos = transform.point(pointer);
    return [
      Math.max(0, Math.min(Math.round(pos.x), imgDimensions.w)),
      Math.max(0, Math.min(Math.round(pos.y), imgDimensions.h)),
    ];
  }, [imgDimensions]);

  // Canvas Stage Click Handling
  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // If clicking on background with select tool, deselect
    if (toolMode === "select") {
      if (e.target === e.target.getStage() || e.target.name() === "bg-image") {
        setSelectedId(null);
      }
      return;
    }

    const pos = getStageRelativePos();
    if (!pos) return;
    const [clickX, clickY] = pos;

    // SAM Smart Click Tool (W): Place point without immediately calling model
    if (toolMode === "sam") {
      const isNegativeModifier = e.evt.button === 2 || e.evt.altKey || e.evt.shiftKey;
      const pointLabel = isNegativeModifier ? 0 : samPointType;

      const newPrompt: PromptPoint = {
        x: clickX,
        y: clickY,
        label: pointLabel,
      };

      setSamPoints((prev) => [...prev, newPrompt]);
      return;
    }

    // Manual Polygon Draw Tool (P)
    if (toolMode === "polygon") {
      if (drawPoints.length >= 3) {
        const [startX, startY] = drawPoints[0];
        const dist = Math.hypot(clickX - startX, clickY - startY);
        if (dist < 18 / stageScale) {
          finishDrawingPolygon();
          return;
        }
      }
      setDrawPoints((prev) => [...prev, [clickX, clickY]]);
    }
  };

  // Stage Mouse Move for drawing preview
  const handleStageMouseMove = () => {
    if (toolMode === "polygon" && drawPoints.length > 0) {
      const pos = getStageRelativePos();
      setMouseCanvasPos(pos);
    }
  };

  // Run SAM Model on currently placed points
  const handleRunSam = useCallback(async () => {
    if (samPoints.length === 0 || isSamLoading) return;
    setIsSamLoading(true);
    try {
      const res = await promptSegment(imageName, samPoints);
      if (res && res.polygon && res.polygon.length >= 3) {
        setSamPendingPolygon(res.polygon);
      } else {
        alert("SAM was unable to trace a confident coral boundary. Try placing additional positive points inside the coral colony.");
      }
    } catch (err: any) {
      console.error("SAM Prompt segmentation error:", err);
      alert(`SAM segmentation error: ${err?.message || err}`);
    } finally {
      setIsSamLoading(false);
    }
  }, [imageName, samPoints, isSamLoading]);

  // Commit SAM mask as coral instance
  const handleCommitSamMask = () => {
    if (!samPendingPolygon || samPendingPolygon.length < 3) return;
    const metrics = calculatePolygonArea(samPendingPolygon);

    const nextId =
      segments.length > 0 ? Math.max(...segments.map((s) => s.id)) + 1 : 1;
    const colorHex = DISTINCT_COLORS[(nextId - 1) % DISTINCT_COLORS.length];

    const newSegment: EditableSegment = {
      id: nextId,
      id_str: `#${nextId}`,
      genus: formGenus,
      growth_form: formGrowthForm,
      condition: formCondition,
      color_hex: colorHex,
      polygon: samPendingPolygon,
      area_px: metrics.area_px,
      area_pct: metrics.area_pct,
      visible: true,
      is_human_annotated: true,
    };

    const nextList = [...segments, newSegment];
    setSegments(nextList);
    segmentsRef.current = nextList;
    pushHistory(nextList);
    setSelectedId(nextId);

    // Reset SAM state
    setSamPoints([]);
    setSamPendingPolygon(null);
    setToolMode("select");

    // Autosave immediately
    triggerDebouncedAutoSave(nextList);
  };

  // Finish manual polygon
  const finishDrawingPolygon = () => {
    if (drawPoints.length < 3) {
      setDrawPoints([]);
      setMouseCanvasPos(null);
      return;
    }

    const closedPoly: [number, number][] = [...drawPoints, drawPoints[0]];
    const metrics = calculatePolygonArea(closedPoly);

    const nextId =
      segments.length > 0 ? Math.max(...segments.map((s) => s.id)) + 1 : 1;
    const colorHex = DISTINCT_COLORS[(nextId - 1) % DISTINCT_COLORS.length];

    const newSegment: EditableSegment = {
      id: nextId,
      id_str: `#${nextId}`,
      genus: formGenus,
      growth_form: formGrowthForm,
      condition: formCondition,
      color_hex: colorHex,
      polygon: closedPoly,
      area_px: metrics.area_px,
      area_pct: metrics.area_pct,
      visible: true,
      is_human_annotated: true,
    };

    const nextList = [...segments, newSegment];
    setSegments(nextList);
    segmentsRef.current = nextList;
    pushHistory(nextList);
    setSelectedId(nextId);

    setDrawPoints([]);
    setMouseCanvasPos(null);
    setToolMode("select");

    triggerDebouncedAutoSave(nextList);
  };

  // Vertex Drag handler for Selected Polygon
  const handleVertexDrag = (
    segId: number,
    vertexIdx: number,
    newPos: { x: number; y: number }
  ) => {
    const updated = segments.map((s) => {
      if (s.id !== segId) return s;
      const poly = [...s.polygon];
      const boundedX = Math.max(0, Math.min(Math.round(newPos.x), imgDimensions.w));
      const boundedY = Math.max(0, Math.min(Math.round(newPos.y), imgDimensions.h));
      poly[vertexIdx] = [boundedX, boundedY];

      // If dragging start or end of closed loop, keep both synced
      if (vertexIdx === 0 && poly.length > 1) {
        poly[poly.length - 1] = [boundedX, boundedY];
      } else if (vertexIdx === poly.length - 1 && poly.length > 1) {
        poly[0] = [boundedX, boundedY];
      }

      const metrics = calculatePolygonArea(poly);
      return {
        ...s,
        polygon: poly as [number, number][],
        area_px: metrics.area_px,
        area_pct: metrics.area_pct,
        is_human_annotated: true,
      };
    });

    setSegments(updated);
    segmentsRef.current = updated;
  };

  const handleVertexDragEnd = () => {
    pushHistory(segmentsRef.current);
    triggerDebouncedAutoSave(segmentsRef.current);
  };

  // Add vertex on line segment double click
  const handleAddVertexOnEdge = (
    segId: number,
    insertIdx: number,
    clickPt: [number, number]
  ) => {
    const updated = segments.map((s) => {
      if (s.id !== segId) return s;
      const poly = [...s.polygon];
      poly.splice(insertIdx + 1, 0, clickPt);
      const metrics = calculatePolygonArea(poly);
      return {
        ...s,
        polygon: poly as [number, number][],
        area_px: metrics.area_px,
        area_pct: metrics.area_pct,
        is_human_annotated: true,
      };
    });
    setSegments(updated);
    segmentsRef.current = updated;
    pushHistory(updated);
    triggerDebouncedAutoSave(updated);
  };

  // Delete vertex (double-click vertex anchor)
  const handleDeleteVertex = (segId: number, vertexIdx: number) => {
    const target = segments.find((s) => s.id === segId);
    if (!target || target.polygon.length <= 4) return; // Keep at least a triangle

    const updated = segments.map((s) => {
      if (s.id !== segId) return s;
      const poly = [...s.polygon];
      poly.splice(vertexIdx, 1);
      if (poly[0] !== poly[poly.length - 1]) {
        poly[poly.length - 1] = poly[0];
      }
      const metrics = calculatePolygonArea(poly);
      return {
        ...s,
        polygon: poly as [number, number][],
        area_px: metrics.area_px,
        area_pct: metrics.area_pct,
        is_human_annotated: true,
      };
    });
    setSegments(updated);
    segmentsRef.current = updated;
    pushHistory(updated);
    triggerDebouncedAutoSave(updated);
  };

  // Delete whole selected coral mask
  const handleDeleteSelected = () => {
    if (selectedId === null) return;
    const nextList = segments.filter((s) => s.id !== selectedId);
    setSegments(nextList);
    segmentsRef.current = nextList;
    pushHistory(nextList);
    setSelectedId(nextList.length > 0 ? nextList[0].id : null);
    triggerDebouncedAutoSave(nextList);
  };

  // Update genus, growth form, condition of selected coral
  const handleUpdateSelectedMetadata = (field: string, value: any) => {
    if (selectedId === null) return;
    const updated = segments.map((s) => {
      if (s.id !== selectedId) return s;
      return {
        ...s,
        [field]: value,
        is_human_annotated: true,
      };
    });
    setSegments(updated);
    segmentsRef.current = updated;
    pushHistory(updated);
    triggerDebouncedAutoSave(updated);
  };

  // Toggle coral visibility
  const handleToggleVisibility = (id: number) => {
    setSegments((prev) =>
      prev.map((s) => (s.id === id ? { ...s, visible: !s.visible } : s))
    );
  };

  // Comprehensive Keyboard Shortcuts (Undo, Redo, Save, SAM, Tools)
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement as HTMLElement;
      const isInput =
        activeEl && ["INPUT", "TEXTAREA", "SELECT"].includes(activeEl.tagName);

      const key = e.key.toLowerCase();
      const isModifier = e.ctrlKey || e.metaKey;

      // Allow Ctrl+S / Cmd+S even inside input fields
      if (isModifier && key === "s") {
        e.preventDefault();
        handleExplicitSave();
        return;
      }

      // If user is actively typing in a form input, avoid intercepting single-letter hotkeys
      if (isInput) return;

      // 1. Undo: Ctrl+Z or Cmd+Z
      if (isModifier && key === "z" && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
        return;
      }

      // 2. Redo: Cmd+Shift+Z, Ctrl+Shift+Z, or Ctrl+Y / Cmd+Y
      if ((isModifier && e.shiftKey && key === "z") || (isModifier && key === "y")) {
        e.preventDefault();
        handleRedo();
        return;
      }

      // 3. SAM specific flow shortcuts
      if (toolMode === "sam") {
        if (e.key === "Enter") {
          e.preventDefault();
          if (samPendingPolygon) {
            handleCommitSamMask();
          } else if (samPoints.length > 0) {
            handleRunSam();
          }
          return;
        }

        if (e.key === " " && !samPendingPolygon && samPoints.length > 0) {
          e.preventDefault();
          handleRunSam();
          return;
        }

        if ((key === "backspace" || key === "delete") && !samPendingPolygon && samPoints.length > 0) {
          e.preventDefault();
          setSamPoints((prev) => prev.slice(0, -1));
          return;
        }
      }

      // 4. Manual Polygon shortcuts
      if (toolMode === "polygon" && drawPoints.length > 0) {
        if (e.key === "Enter" && drawPoints.length >= 3) {
          e.preventDefault();
          finishDrawingPolygon();
          return;
        }
        if (key === "backspace" || key === "delete") {
          e.preventDefault();
          setDrawPoints((prev) => prev.slice(0, -1));
          return;
        }
      }

      // 5. Tool switches
      if (key === "v") {
        setToolMode("select");
        setSamPoints([]);
        setSamPendingPolygon(null);
        setDrawPoints([]);
      } else if (key === "w") {
        setToolMode("sam");
        setDrawPoints([]);
      } else if (key === "p") {
        setToolMode("polygon");
        setSamPoints([]);
        setSamPendingPolygon(null);
      } else if (key === "delete" || key === "backspace") {
        if (selectedId !== null) {
          handleDeleteSelected();
        }
      } else if (e.key === "Escape") {
        if (toolMode === "polygon" && drawPoints.length > 0) {
          setDrawPoints([]);
          setMouseCanvasPos(null);
        } else if (toolMode === "sam" && samPendingPolygon) {
          setSamPendingPolygon(null);
        } else if (toolMode === "sam" && samPoints.length > 0) {
          setSamPoints([]);
        } else if (selectedId !== null) {
          setSelectedId(null);
        } else {
          onClose();
        }
      } else if (key === "0" || key === "f") {
        handleFitToScreen();
      } else if (key === "[" && hasPrev && onNavigatePrev) {
        onNavigatePrev();
      } else if (key === "]" && hasNext && onNavigateNext) {
        onNavigateNext();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    isOpen,
    toolMode,
    samPoints,
    samPendingPolygon,
    drawPoints,
    selectedId,
    handleExplicitSave,
    handleUndo,
    handleRedo,
    handleRunSam,
    handleFitToScreen,
    hasPrev,
    hasNext,
    onNavigatePrev,
    onNavigateNext,
    onClose,
  ]);

  if (!isOpen) return null;

  const activeSegment = segments.find((s) => s.id === selectedId) || null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-slate-900/60 backdrop-blur-xs select-none overflow-hidden animate-fade-in font-sans">
      {/* 1. Header Toolbar (ReefVision Studio Brand Theme) */}
      <header className="h-14 bg-white border-b border-slate-200 px-4 flex items-center justify-between gap-4 shrink-0 shadow-xs z-20">
        {/* Left: Branding, Filename, Resolution, Save Status */}
        <div className="flex items-center gap-3 min-w-0">
          <Badge
            variant="outline"
            className="text-[10px] font-mono font-bold uppercase tracking-wider bg-teal-50 text-teal-800 border-teal-200 px-2 py-0.5 shrink-0"
          >
            Human Annotation
          </Badge>

          <div className="flex items-center gap-2 min-w-0">
            <span className="text-sm font-semibold text-slate-900 truncate max-w-[200px] md:max-w-[320px]">
              {imageName}
            </span>
            <span className="text-[11px] font-mono text-slate-500 shrink-0">
              {imageResolution && imageResolution !== "0x0"
                ? imageResolution
                : `${imgDimensions.w}x${imgDimensions.h}`}
            </span>
          </div>

          {/* Live Autosave & Save Status Indicator */}
          <div className="hidden sm:flex items-center shrink-0">
            {saveStatus === "saving" ? (
              <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border bg-blue-50 border-blue-200 text-blue-800">
                <Loader2 size={12} className="animate-spin text-blue-600" />
                <span>Autosaving...</span>
              </div>
            ) : hasUnsavedChanges || saveStatus === "unsaved" ? (
              <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border bg-amber-50 border-amber-200 text-amber-800">
                <span className="size-2 rounded-full bg-amber-500 animate-pulse" />
                <span>Unsaved changes</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border bg-emerald-50 border-emerald-200 text-emerald-800">
                <CheckCircle2 size={12} className="text-emerald-600" />
                <span>
                  {lastSavedTime ? `Saved at ${lastSavedTime}` : "Saved"}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Center: Tools Selector (CVAT / Roboflow Style) */}
        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
          <button
            type="button"
            onClick={() => {
              setToolMode("select");
              setSamPoints([]);
              setSamPendingPolygon(null);
              setDrawPoints([]);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              toolMode === "select"
                ? "bg-[#0d7c85] text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-white"
            }`}
            title="Select & Edit Vertices (Hotkey: V)"
          >
            <MousePointer size={14} />
            <span>Select (V)</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setToolMode("sam");
              setDrawPoints([]);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              toolMode === "sam"
                ? "bg-[#0d7c85] text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-white"
            }`}
            title="SAM Smart Click Model (Hotkey: W)"
          >
            {isSamLoading ? (
              <Loader2 size={14} className="animate-spin text-teal-200" />
            ) : (
              <Wand2 size={14} />
            )}
            <span>SAM Smart Click (W)</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setToolMode("polygon");
              setSamPoints([]);
              setSamPendingPolygon(null);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              toolMode === "polygon"
                ? "bg-[#0d7c85] text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-white"
            }`}
            title="Manual Polygon Boundary Draw (Hotkey: P)"
          >
            <Pentagon size={14} />
            <span>Polygon (P)</span>
          </button>
        </div>

        {/* Right: Undo/Redo, Navigation, Save & Close */}
        <div className="flex items-center gap-2">
          {/* Undo / Redo */}
          <div className="hidden md:flex items-center gap-1 border-r border-slate-200 pr-2">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              disabled={historyIdx <= 0}
              onClick={handleUndo}
              className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
              title="Undo (Ctrl+Z / Cmd+Z)"
            >
              <Undo2 size={15} />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              disabled={historyIdx >= historyLength - 1}
              onClick={handleRedo}
              className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
              title="Redo (Ctrl+Y / Cmd+Shift+Z)"
            >
              <Redo2 size={15} />
            </Button>
          </div>

          {/* Quick Pagination */}
          {(hasPrev || hasNext) && (
            <div className="flex items-center gap-1 border-r border-slate-200 pr-2">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                disabled={!hasPrev}
                onClick={onNavigatePrev}
                className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
                title="Previous Image ([)"
              >
                <ChevronLeft size={16} />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                disabled={!hasNext}
                onClick={onNavigateNext}
                className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-40"
                title="Next Image (])"
              >
                <ChevronRight size={16} />
              </Button>
            </div>
          )}

          {/* Save Button */}
          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={isSaving}
            onClick={handleExplicitSave}
            className="h-8 px-3 text-xs font-semibold gap-1.5 bg-[#0d7c85] hover:bg-[#0b6b73] text-white shadow-xs cursor-pointer"
            title="Save Annotations (Ctrl+S / Cmd+S)"
          >
            {isSaving ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
            <span>Save</span>
            <kbd className="hidden md:inline-block ml-0.5 text-[10px] bg-white/20 px-1 py-0.2 rounded font-mono">
              ⌘S
            </kbd>
          </Button>

          {/* Close */}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-slate-500 hover:text-slate-900 hover:bg-slate-100 ml-1"
            title="Close Editor (Esc)"
          >
            <X size={18} />
          </Button>
        </div>
      </header>

      {/* 2. Main Studio Body: Viewport Canvas + Right Inspector */}
      <div className="flex flex-1 min-h-0 overflow-hidden relative">
        {/* Canvas Stage Viewport (High-Contrast Neutral Slate #0f172a, matching Lightbox) */}
        <div
          ref={containerRef}
          className="flex-1 min-w-0 h-full relative bg-[#0f172a] overflow-hidden select-none"
        >
          {htmlImage && (
            <Stage
              ref={stageRef}
              width={containerRef.current?.clientWidth || 800}
              height={containerRef.current?.clientHeight || 600}
              scaleX={stageScale}
              scaleY={stageScale}
              x={stagePos.x}
              y={stagePos.y}
              draggable={toolMode === "select" && !selectedId}
              onDragEnd={(e) => {
                if (e.target === stageRef.current) {
                  setStagePos({ x: e.target.x(), y: e.target.y() });
                }
              }}
              onWheel={handleWheel}
              onClick={handleStageClick}
              onMouseMove={handleStageMouseMove}
              style={{ cursor: toolMode === "select" ? "default" : "crosshair" }}
            >
              {/* Background Reef Image */}
              <Layer>
                <KonvaImage
                  name="bg-image"
                  image={htmlImage}
                  width={imgDimensions.w}
                  height={imgDimensions.h}
                />
              </Layer>

              {/* Polygons & Annotations Layer */}
              <Layer>
                {segments.map((seg) => {
                  if (!seg.visible || seg.polygon.length < 3) return null;
                  const isSelected = seg.id === selectedId;
                  const flatPoints = seg.polygon.flatMap((pt) => [pt[0], pt[1]]);
                  const [cx, cy] = seg.polygon[0];

                  return (
                    <Group key={seg.id}>
                      {/* Polygon fill & stroke */}
                      <Line
                        points={flatPoints}
                        closed={true}
                        fill={seg.color_hex}
                        opacity={isSelected ? Math.min(maskOpacity + 0.2, 0.85) : maskOpacity}
                        stroke={isSelected ? "#ffffff" : seg.color_hex}
                        strokeWidth={isSelected ? 3 / stageScale : 2 / stageScale}
                        shadowColor={isSelected ? "#000000" : undefined}
                        shadowBlur={isSelected ? 6 : 0}
                        onClick={(e) => {
                          e.cancelBubble = true;
                          setSelectedId(seg.id);
                        }}
                        onDblClick={(e) => {
                          if (isSelected && toolMode === "select") {
                            e.cancelBubble = true;
                            const stage = stageRef.current;
                            if (stage) {
                              const transform = stage.getAbsoluteTransform().copy().invert();
                              const pointer = stage.getPointerPosition();
                              if (pointer) {
                                const pt = transform.point(pointer);
                                handleAddVertexOnEdge(seg.id, 0, [Math.round(pt.x), Math.round(pt.y)]);
                              }
                            }
                          }
                        }}
                      />

                      {/* Floating ID badge */}
                      <Group x={cx} y={cy}>
                        <Rect
                          x={-24 / stageScale}
                          y={-14 / stageScale}
                          width={48 / stageScale}
                          height={16 / stageScale}
                          fill={isSelected ? "#0d7c85" : "#0f172a"}
                          stroke={isSelected ? "#ffffff" : seg.color_hex}
                          strokeWidth={1 / stageScale}
                          cornerRadius={3 / stageScale}
                        />
                        <KonvaText
                          x={-24 / stageScale}
                          y={-12 / stageScale}
                          width={48 / stageScale}
                          text={seg.id_str}
                          fontSize={10 / stageScale}
                          fontStyle="bold"
                          fontFamily="monospace"
                          fill="#ffffff"
                          align="center"
                        />
                      </Group>

                      {/* Interactive Vertex Handles when Selected */}
                      {isSelected &&
                        toolMode === "select" &&
                        seg.polygon.map((pt, vIdx) => {
                          if (vIdx === seg.polygon.length - 1) return null;
                          return (
                            <Circle
                              key={vIdx}
                              x={pt[0]}
                              y={pt[1]}
                              radius={5.5 / stageScale}
                              fill="#ffffff"
                              stroke="#0f172a"
                              strokeWidth={1.5 / stageScale}
                              draggable
                              onDragMove={(e) => {
                                handleVertexDrag(seg.id, vIdx, {
                                  x: e.target.x(),
                                  y: e.target.y(),
                                });
                              }}
                              onDragEnd={() => {
                                handleVertexDragEnd();
                              }}
                              onDblClick={(e) => {
                                e.cancelBubble = true;
                                handleDeleteVertex(seg.id, vIdx);
                              }}
                            />
                          );
                        })}
                    </Group>
                  );
                })}

                {/* SAM Prompt Points Pins (Green + for Coral, Red - for Background) */}
                {samPoints.map((pt, idx) => {
                  const isPositive = pt.label === 1;
                  const pinColor = isPositive ? "#10b981" : "#ef4444";
                  return (
                    <Group key={idx} x={pt.x} y={pt.y}>
                      <Circle
                        radius={10 / stageScale}
                        fill={pinColor}
                        opacity={0.25}
                      />
                      <Circle
                        radius={7 / stageScale}
                        fill={pinColor}
                        stroke="#ffffff"
                        strokeWidth={1.5 / stageScale}
                        shadowColor="#000000"
                        shadowBlur={4}
                      />
                      <KonvaText
                        x={-6 / stageScale}
                        y={-5 / stageScale}
                        width={12 / stageScale}
                        text={isPositive ? "+" : "−"}
                        fontSize={10 / stageScale}
                        fontStyle="bold"
                        fontFamily="sans-serif"
                        fill="#ffffff"
                        align="center"
                      />
                    </Group>
                  );
                })}

                {/* SAM Predicted Polygon Preview */}
                {samPendingPolygon && samPendingPolygon.length >= 3 && (
                  <Line
                    points={samPendingPolygon.flatMap((p) => [p[0], p[1]])}
                    closed={true}
                    fill="#10b981"
                    opacity={0.45}
                    stroke="#ffffff"
                    strokeWidth={2.5 / stageScale}
                    dash={[4 / stageScale, 4 / stageScale]}
                  />
                )}

                {/* Manual Polygon Drawing in Progress */}
                {drawPoints.length > 0 && (
                  <Group>
                    <Line
                      points={[
                        ...drawPoints.flatMap((p) => [p[0], p[1]]),
                        ...(mouseCanvasPos ? [mouseCanvasPos[0], mouseCanvasPos[1]] : []),
                      ]}
                      closed={false}
                      stroke="#00e5ff"
                      strokeWidth={2 / stageScale}
                      dash={[3 / stageScale, 3 / stageScale]}
                    />
                    {drawPoints.map((p, idx) => (
                      <Circle
                        key={idx}
                        x={p[0]}
                        y={p[1]}
                        radius={idx === 0 ? 6 / stageScale : 4 / stageScale}
                        fill={idx === 0 ? "#ffd600" : "#00e5ff"}
                        stroke="#ffffff"
                        strokeWidth={1.5 / stageScale}
                      />
                    ))}
                  </Group>
                )}
              </Layer>
            </Stage>
          )}

          {/* Floating Canvas Controls (Zoom, Fit, Opacity) */}
          <div className="absolute bottom-4 left-4 z-10 flex items-center gap-2 bg-white/95 backdrop-blur-md p-1.5 rounded-lg border border-slate-200 shadow-md text-xs text-slate-700">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => {
                const newScale = Math.max(0.2, Number((stageScale * 0.85).toFixed(2)));
                setStageScale(newScale);
              }}
              className="h-7 w-7 text-slate-600 hover:text-slate-900"
              title="Zoom Out (-)"
            >
              <ZoomOut size={13} />
            </Button>

            <span className="font-mono text-[11px] text-slate-700 px-1 min-w-[42px] text-center font-semibold">
              {Math.round(stageScale * 100)}%
            </span>

            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => {
                const newScale = Math.min(8.0, Number((stageScale * 1.18).toFixed(2)));
                setStageScale(newScale);
              }}
              className="h-7 w-7 text-slate-600 hover:text-slate-900"
              title="Zoom In (+)"
            >
              <ZoomIn size={13} />
            </Button>

            <div className="h-4 w-px bg-slate-200 mx-1" />

            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleFitToScreen}
              className="h-7 px-2 text-[11px] text-slate-600 hover:text-slate-900 gap-1"
              title="Fit to Screen (F / 0)"
            >
              <RotateCcw size={12} />
              <span>Fit (F)</span>
            </Button>

            <div className="h-4 w-px bg-slate-200 mx-1" />

            {/* Opacity Slider */}
            <div className="flex items-center gap-1.5 px-2 text-[11px] text-slate-600">
              <Sliders size={11} />
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={maskOpacity}
                onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                className="w-16 accent-teal-600 cursor-pointer"
                title="Mask Opacity"
              />
            </div>
          </div>

          {/* Enhanced Multi-Point SAM Flow Top Banner */}
          {toolMode === "sam" && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex flex-col sm:flex-row items-center gap-2.5 bg-white/95 text-slate-900 backdrop-blur-md px-4 py-2.5 rounded-xl border border-slate-200 shadow-xl animate-in fade-in slide-in-from-top-3">
              {/* Left: Mode Title & Info */}
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-teal-50 text-teal-700">
                  <Wand2 size={16} />
                </div>
                <div className="flex flex-col">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-bold text-slate-800">SAM Smart Click</span>
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-teal-50 text-teal-700 border-teal-200">
                      Interactive
                    </Badge>
                  </div>
                  <span className="text-[11px] text-slate-500">
                    {samPendingPolygon
                      ? "Colony mask generated. Accept or add points to refine."
                      : samPoints.length === 0
                      ? "Click coral to place points, then run model."
                      : `${samPoints.length} point${samPoints.length > 1 ? "s" : ""} placed (${samPoints.filter((p) => p.label === 1).length} +, ${samPoints.filter((p) => p.label === 0).length} −)`}
                  </span>
                </div>
              </div>

              <div className="h-6 w-px bg-slate-200 hidden sm:block" />

              {/* Point Type Selector (+ / −) */}
              {!samPendingPolygon && (
                <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
                  <button
                    type="button"
                    onClick={() => setSamPointType(1)}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-semibold cursor-pointer transition-all ${
                      samPointType === 1
                        ? "bg-emerald-600 text-white shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                    title="Add Positive Foreground Points"
                  >
                    <span className="font-bold">+</span>
                    <span>Coral</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setSamPointType(0)}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-semibold cursor-pointer transition-all ${
                      samPointType === 0
                        ? "bg-red-600 text-white shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                    title="Add Negative Background Points (or hold Alt/Shift)"
                  >
                    <span className="font-bold">−</span>
                    <span>Background</span>
                  </button>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                {samPendingPolygon ? (
                  <>
                    <Button
                      type="button"
                      variant="default"
                      size="sm"
                      onClick={handleCommitSamMask}
                      className="h-7 px-3 text-xs font-bold bg-[#0d7c85] hover:bg-[#0b6b73] text-white shadow-xs cursor-pointer gap-1"
                    >
                      <Check size={13} />
                      <span>Accept Mask</span>
                      <kbd className="ml-1 text-[10px] bg-white/20 px-1 py-0.2 rounded font-mono">↵ Enter</kbd>
                    </Button>

                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={isSamLoading}
                      onClick={handleRunSam}
                      className="h-7 px-2.5 text-xs text-slate-700 hover:text-slate-900 border-slate-300 hover:bg-slate-50 cursor-pointer gap-1"
                    >
                      {isSamLoading ? <Loader2 size={12} className="animate-spin" /> : <Sparkles size={12} />}
                      <span>Re-run SAM</span>
                      <kbd className="ml-1 text-[10px] bg-slate-200 px-1 py-0.2 rounded font-mono">Space</kbd>
                    </Button>

                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setSamPoints([]);
                        setSamPendingPolygon(null);
                      }}
                      className="h-7 px-2 text-xs text-slate-500 hover:text-slate-800"
                    >
                      Discard
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      type="button"
                      variant="default"
                      size="sm"
                      disabled={samPoints.length === 0 || isSamLoading}
                      onClick={handleRunSam}
                      className="h-7 px-3 text-xs font-bold bg-[#0d7c85] hover:bg-[#0b6b73] text-white shadow-xs cursor-pointer gap-1.5"
                    >
                      {isSamLoading ? (
                        <>
                          <Loader2 size={13} className="animate-spin" />
                          <span>Computing...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles size={13} />
                          <span>Run SAM Model</span>
                          <kbd className="ml-1 text-[10px] bg-white/20 px-1 py-0.2 rounded font-mono">↵ Enter</kbd>
                        </>
                      )}
                    </Button>

                    {samPoints.length > 0 && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => setSamPoints((prev) => prev.slice(0, -1))}
                        className="h-7 px-2 text-xs text-slate-600 hover:text-slate-900 border-slate-300"
                        title="Undo last point (Backspace)"
                      >
                        <Undo2 size={12} />
                      </Button>
                    )}

                    {samPoints.length > 0 && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => setSamPoints([])}
                        className="h-7 px-2 text-xs text-slate-500 hover:text-slate-800"
                      >
                        Clear
                      </Button>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Polygon Drawing In Progress Instruction Banner */}
          {toolMode === "polygon" && drawPoints.length > 0 && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-white/95 text-slate-900 backdrop-blur-md px-4 py-2 rounded-xl border border-slate-200 shadow-xl text-xs">
              <span className="text-slate-700 font-medium">
                Click around coral boundary • Click start point or Close button to finish
              </span>
              <Button
                type="button"
                variant="default"
                size="sm"
                onClick={finishDrawingPolygon}
                className="h-6 px-2.5 text-[11px] font-semibold bg-[#0d7c85] hover:bg-[#0b6b73] text-white"
              >
                Close Polygon
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => {
                  setDrawPoints([]);
                  setMouseCanvasPos(null);
                }}
                className="h-6 px-2 text-[11px] text-slate-500 hover:text-slate-800"
              >
                Cancel
              </Button>
            </div>
          )}
        </div>

        {/* 3. Right Inspector & Corals Catalog Sidebar (Clean Site Theme) */}
        <aside className="w-80 md:w-92 bg-white border-l border-slate-200 flex flex-col shrink-0 h-full overflow-hidden shadow-lg z-10">
          {/* Active Coral Form Card */}
          <div className="p-4 border-b border-slate-200 bg-slate-50/70">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                  {activeSegment ? `Edit Coral ${activeSegment.id_str}` : "New Coral Definition"}
                </span>
                {activeSegment && (
                  <span
                    className="size-2.5 rounded-full ring-1 ring-slate-400"
                    style={{ backgroundColor: activeSegment.color_hex }}
                  />
                )}
              </div>
              <Badge
                variant="outline"
                className="text-[10px] font-mono bg-teal-50 text-teal-800 border-teal-200"
              >
                100% Ground Truth
              </Badge>
            </div>

            <div className="space-y-3">
              {/* Taxon Genus Selector */}
              <div>
                <label className="text-[11px] font-semibold text-slate-700 mb-1 block">
                  Taxon Genus:
                </label>
                <select
                  value={activeSegment ? activeSegment.genus : formGenus}
                  onChange={(e) => {
                    const val = e.target.value;
                    const matched = TAXONOMY_OPTIONS.find((t) => t.genus === val);
                    if (activeSegment) {
                      handleUpdateSelectedMetadata("genus", val);
                      if (matched) {
                        handleUpdateSelectedMetadata("growth_form", matched.form);
                      }
                    } else {
                      setFormGenus(val);
                      if (matched) {
                        setFormGrowthForm(matched.form);
                      }
                    }
                  }}
                  className="w-full h-8 px-2.5 bg-white border border-slate-300 rounded-md text-xs text-slate-800 focus:outline-hidden focus:border-teal-600 focus:ring-1 focus:ring-teal-500 cursor-pointer shadow-2xs"
                >
                  {TAXONOMY_OPTIONS.map((opt) => (
                    <option key={opt.genus} value={opt.genus}>
                      {opt.genus} ({opt.common})
                    </option>
                  ))}
                </select>
              </div>

              {/* Growth Form Selector */}
              <div>
                <label className="text-[11px] font-semibold text-slate-700 mb-1 block">
                  Growth Form:
                </label>
                <select
                  value={activeSegment ? activeSegment.growth_form : formGrowthForm}
                  onChange={(e) => {
                    const val = e.target.value;
                    if (activeSegment) {
                      handleUpdateSelectedMetadata("growth_form", val);
                    } else {
                      setFormGrowthForm(val);
                    }
                  }}
                  className="w-full h-8 px-2.5 bg-white border border-slate-300 rounded-md text-xs text-slate-800 focus:outline-hidden focus:border-teal-600 focus:ring-1 focus:ring-teal-500 cursor-pointer shadow-2xs"
                >
                  {GROWTH_FORMS.map((form) => (
                    <option key={form} value={form}>
                      {form}
                    </option>
                  ))}
                </select>
              </div>

              {/* Condition Toggle (Healthy vs Bleached) */}
              <div>
                <label className="text-[11px] font-semibold text-slate-700 mb-1 block">
                  Health Condition:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      if (activeSegment) {
                        handleUpdateSelectedMetadata("condition", "Healthy");
                      } else {
                        setFormCondition("Healthy");
                      }
                    }}
                    className={`h-7 rounded-md text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                      (activeSegment ? activeSegment.condition : formCondition) === "Healthy"
                        ? "bg-emerald-600 text-white shadow-xs"
                        : "bg-slate-200/80 text-slate-700 hover:bg-slate-300/80"
                    }`}
                  >
                    <span>Healthy</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      if (activeSegment) {
                        handleUpdateSelectedMetadata("condition", "Bleached");
                      } else {
                        setFormCondition("Bleached");
                      }
                    }}
                    className={`h-7 rounded-md text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                      (activeSegment ? activeSegment.condition : formCondition) === "Bleached"
                        ? "bg-red-600 text-white shadow-xs"
                        : "bg-slate-200/80 text-slate-700 hover:bg-slate-300/80"
                    }`}
                  >
                    <span>Bleached</span>
                  </button>
                </div>
              </div>

              {/* Live Area Telemetry */}
              {activeSegment && (
                <div className="pt-2 border-t border-slate-200 flex items-center justify-between text-[11px] bg-white p-2.5 rounded-lg border border-slate-200/70 shadow-2xs">
                  <div className="flex flex-col">
                    <span className="text-slate-500">Calculated Area</span>
                    <span className="font-mono font-semibold text-slate-900">
                      {activeSegment.area_px.toLocaleString()} px²
                    </span>
                  </div>
                  <div className="flex flex-col items-end">
                    <span className="text-slate-500">Reef Coverage</span>
                    <span className="font-mono font-bold text-teal-700">
                      {activeSegment.area_pct}%
                    </span>
                  </div>
                </div>
              )}

              {/* Delete & Deselect Action */}
              {activeSegment && (
                <div className="pt-2 flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleDeleteSelected}
                    className="flex-1 h-7 text-xs text-red-600 border-red-200 bg-red-50 hover:bg-red-100 hover:text-red-700 cursor-pointer"
                  >
                    <Trash2 size={12} className="mr-1" />
                    <span>Delete Mask</span>
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedId(null)}
                    className="h-7 text-xs text-slate-600 hover:text-slate-900"
                  >
                    Deselect
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Corals Catalog List */}
          <div className="flex-1 min-h-0 flex flex-col overflow-hidden bg-white">
            <div className="px-4 py-2.5 border-b border-slate-200 flex items-center justify-between bg-white">
              <span className="text-xs font-bold text-slate-800">
                Instances ({segments.length})
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  setSelectedId(null);
                  setToolMode("sam");
                }}
                className="h-6 px-2 text-[10px] font-semibold text-teal-700 border-teal-300 bg-teal-50 hover:bg-teal-100 gap-1 cursor-pointer"
              >
                <Plus size={11} />
                <span>Add Coral</span>
              </Button>
            </div>

            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {segments.length === 0 ? (
                <div className="p-8 text-center text-xs text-slate-500">
                  No coral annotations on this image yet. Click{" "}
                  <strong className="text-teal-700">SAM Smart Click</strong> or{" "}
                  <strong className="text-cyan-700">Polygon</strong> above to segment a colony.
                </div>
              ) : (
                segments.map((seg) => {
                  const isSelected = seg.id === selectedId;
                  return (
                    <div
                      key={seg.id}
                      onClick={() => setSelectedId(seg.id)}
                      className={`flex items-center justify-between p-2 rounded-lg text-xs transition-all cursor-pointer ${
                        isSelected
                          ? "bg-teal-50/80 border border-teal-500/60 shadow-2xs"
                          : "hover:bg-slate-50 border border-slate-200/80"
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span
                          className="size-2.5 rounded-full ring-1 ring-slate-300 shrink-0"
                          style={{ backgroundColor: seg.color_hex }}
                        />
                        <div className="flex flex-col min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono font-bold text-slate-900">
                              {seg.id_str}
                            </span>
                            <span className="font-semibold text-slate-800 truncate">
                              {seg.genus}
                            </span>
                          </div>
                          <span className="text-[10px] text-slate-500 truncate">
                            {seg.growth_form} • {seg.area_pct}%
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-1 shrink-0">
                        <Badge
                          variant="outline"
                          className={`text-[9px] px-1.5 py-0 font-semibold ${
                            seg.condition === "Bleached"
                              ? "bg-red-50 text-red-700 border-red-200"
                              : "bg-emerald-50 text-emerald-700 border-emerald-200"
                          }`}
                        >
                          {seg.condition}
                        </Badge>

                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleToggleVisibility(seg.id);
                          }}
                          className="p-1 text-slate-400 hover:text-slate-800"
                          title="Toggle visibility"
                        >
                          {seg.visible ? <Eye size={13} /> : <EyeOff size={13} />}
                        </button>

                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedId(seg.id);
                            handleDeleteSelected();
                          }}
                          className="p-1 text-slate-400 hover:text-red-600"
                          title="Delete"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Quick Keyboard Shortcuts Footer */}
          <footer className="p-3 bg-slate-50 border-t border-slate-200 text-[10px] text-slate-600 grid grid-cols-2 gap-2">
            <div>
              <kbd className="px-1.5 py-0.5 bg-white border border-slate-300 rounded font-mono text-slate-800 shadow-2xs">V</kbd>{" "}
              Select & Edit
            </div>
            <div>
              <kbd className="px-1.5 py-0.5 bg-white border border-slate-300 rounded font-mono text-slate-800 shadow-2xs">W</kbd>{" "}
              SAM Smart Click
            </div>
            <div>
              <kbd className="px-1.5 py-0.5 bg-white border border-slate-300 rounded font-mono text-slate-800 shadow-2xs">P</kbd>{" "}
              Polygon Tool
            </div>
            <div>
              <kbd className="px-1.5 py-0.5 bg-white border border-slate-300 rounded font-mono text-slate-800 shadow-2xs">⌘Z</kbd>{" "}
              Undo Action
            </div>
            <div>
              <kbd className="px-1.5 py-0.5 bg-white border border-slate-300 rounded font-mono text-slate-800 shadow-2xs">⌘S</kbd>{" "}
              Save Changes
            </div>
            <div>
              <kbd className="px-1.5 py-0.5 bg-white border border-slate-300 rounded font-mono text-slate-800 shadow-2xs">[ / ]</kbd>{" "}
              Prev / Next
            </div>
          </footer>
        </aside>
      </div>
    </div>
  );
};
