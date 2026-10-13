import { useState, useRef, useEffect, useCallback } from "react";
import { EditableSegment, SaveStatus } from "./types";
import { CoralSegment, SummaryStats, HealthSummary } from "../../types";
import {
  cloneSegments,
  getUniquePolygonPoints,
  ensureClosedPolygon,
  calculatePolygonArea,
} from "./geometry";
import { DISTINCT_COLORS } from "./constants";
import { getSegmentPolygons, saveAnnotations } from "../../services/api";

interface UseSegmentManagerProps {
  imageName: string;
  imageSrc: string;
  initialSegments?: CoralSegment[];
  imgDimensions: { w: number; h: number };
  onSaveSuccess?: (data: {
    summary: SummaryStats;
    segments: CoralSegment[];
    health_summary: HealthSummary;
    overlay_base64: string;
  }) => void;
  resetHistory: (list: EditableSegment[]) => void;
  pushHistory: (list: EditableSegment[]) => void;
}

export function useSegmentManager({
  imageName,
  imageSrc,
  initialSegments = [],
  imgDimensions,
  onSaveSuccess,
  resetHistory,
  pushHistory,
}: UseSegmentManagerProps) {
  const [segments, setSegments] = useState<EditableSegment[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedVertexIdx, setSelectedVertexIdx] = useState<number | null>(null);
  const [editorNotice, setEditorNotice] = useState<string | null>(null);

  // Save / Dirty status
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saved");
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState<boolean>(false);
  const [lastSavedTime, setLastSavedTime] = useState<string | null>(null);

  const segmentsRef = useRef<EditableSegment[]>([]);
  const hasUnsavedRef = useRef<boolean>(false);
  const isSavingRef = useRef<boolean>(false);
  const initialSegmentsRef = useRef<CoralSegment[]>(initialSegments);

  // Keep segmentsRef in sync
  useEffect(() => {
    segmentsRef.current = segments;
  }, [segments]);

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
        resetHistory(cloned);
        hasUnsavedRef.current = false;
        setHasUnsavedChanges(false);
        setSaveStatus("saved");
        setSelectedVertexIdx(null);
        if (cloned.length > 0) {
          setSelectedId(cloned[0].id);
        }
        return;
      }
    } catch {
      // Fallback to initialSegments if available
    }

    const fallbackSource = initialSegmentsRef.current || [];
    if (fallbackSource.length > 0) {
      const fallback: EditableSegment[] = fallbackSource.map((s, idx) => ({
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
      resetHistory(cloned);
      hasUnsavedRef.current = false;
      setHasUnsavedChanges(false);
      setSaveStatus("saved");
      setSelectedVertexIdx(null);
      if (cloned.length > 0) {
        setSelectedId(cloned[0].id);
      }
    } else {
      setSegments([]);
      segmentsRef.current = [];
      resetHistory([]);
      hasUnsavedRef.current = false;
      setHasUnsavedChanges(false);
      setSaveStatus("saved");
      setSelectedId(null);
      setSelectedVertexIdx(null);
    }
  }, [imageName, resetHistory]);

  // Save Annotations function (Unified for manual Cmd+S and periodic 20s auto-save)
  const executeSave = useCallback(
    async (isManual: boolean = false) => {
      if (isSavingRef.current) return;
      if (!hasUnsavedRef.current && !isManual) return;

      isSavingRef.current = true;
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

        const res = await saveAnnotations(imageName, payload, imageSrc);
        hasUnsavedRef.current = false;
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
        console.warn("Save warning:", err);
        setSaveStatus("unsaved");
        if (isManual) {
          alert(`Save failed: ${err?.message || err}`);
        }
      } finally {
        isSavingRef.current = false;
        setIsSaving(false);
      }
    },
    [imageName, imageSrc, onSaveSuccess]
  );

  // Add a newly created segment (from SAM or manual drawing)
  const addSegment = useCallback(
    (newSegment: EditableSegment) => {
      const nextList = [...segmentsRef.current, newSegment];
      setSegments(nextList);
      segmentsRef.current = nextList;
      pushHistory(nextList);
      setSelectedId(newSegment.id);
      setSelectedVertexIdx(null);
    },
    [pushHistory]
  );

  // Vertex Drag handler for Selected Polygon
  const handleVertexDrag = useCallback(
    (segId: number, vertexIdx: number, newPos: { x: number; y: number }) => {
      const target = segmentsRef.current.find((s) => s.id === segId);
      if (!target) return;

      const uniquePts = getUniquePolygonPoints(target.polygon);
      if (vertexIdx < 0 || vertexIdx >= uniquePts.length) return;

      const boundedX = Math.max(0, Math.min(Math.round(newPos.x), imgDimensions.w));
      const boundedY = Math.max(0, Math.min(Math.round(newPos.y), imgDimensions.h));

      const updatedPts = [...uniquePts];
      updatedPts[vertexIdx] = [boundedX, boundedY];
      const closed = ensureClosedPolygon(updatedPts);
      const metrics = calculatePolygonArea(closed, imgDimensions.w, imgDimensions.h);

      const updated = segmentsRef.current.map((s) => {
        if (s.id !== segId) return s;
        return {
          ...s,
          polygon: closed,
          area_px: metrics.area_px,
          area_pct: metrics.area_pct,
          is_human_annotated: true,
        };
      });

      setSegments(updated);
      segmentsRef.current = updated;
    },
    [imgDimensions]
  );

  const handleVertexDragEnd = useCallback(() => {
    pushHistory(segmentsRef.current);
  }, [pushHistory]);

  // Add vertex on line segment / midpoint handle click
  const handleAddVertexOnEdge = useCallback(
    (segId: number, edgeIdx: number, clickPt: [number, number]) => {
      const target = segmentsRef.current.find((s) => s.id === segId);
      if (!target) return;

      const uniquePts = getUniquePolygonPoints(target.polygon);
      if (edgeIdx < 0 || edgeIdx >= uniquePts.length) return;

      const boundedX = Math.max(0, Math.min(Math.round(clickPt[0]), imgDimensions.w));
      const boundedY = Math.max(0, Math.min(Math.round(clickPt[1]), imgDimensions.h));

      const updatedPts = [...uniquePts];
      updatedPts.splice(edgeIdx + 1, 0, [boundedX, boundedY]);
      const closed = ensureClosedPolygon(updatedPts);
      const metrics = calculatePolygonArea(closed, imgDimensions.w, imgDimensions.h);

      const updated = segmentsRef.current.map((s) => {
        if (s.id !== segId) return s;
        return {
          ...s,
          polygon: closed,
          area_px: metrics.area_px,
          area_pct: metrics.area_pct,
          is_human_annotated: true,
        };
      });

      setSegments(updated);
      segmentsRef.current = updated;
      pushHistory(updated);
      setSelectedVertexIdx(edgeIdx + 1);
      setEditorNotice("Point added to polygon edge.");
    },
    [imgDimensions, pushHistory]
  );

  // Delete vertex
  const handleDeleteVertex = useCallback(
    (segId: number, vertexIdx: number) => {
      const target = segmentsRef.current.find((s) => s.id === segId);
      if (!target) return;

      const uniquePts = getUniquePolygonPoints(target.polygon);
      if (uniquePts.length <= 3) {
        setEditorNotice("Cannot delete point: a coral polygon must have at least 3 vertices.");
        return;
      }
      if (vertexIdx < 0 || vertexIdx >= uniquePts.length) return;

      const updatedPts = [...uniquePts];
      updatedPts.splice(vertexIdx, 1);
      const closed = ensureClosedPolygon(updatedPts);
      const metrics = calculatePolygonArea(closed, imgDimensions.w, imgDimensions.h);

      const updated = segmentsRef.current.map((s) => {
        if (s.id !== segId) return s;
        return {
          ...s,
          polygon: closed,
          area_px: metrics.area_px,
          area_pct: metrics.area_pct,
          is_human_annotated: true,
        };
      });

      setSegments(updated);
      segmentsRef.current = updated;
      pushHistory(updated);
      setSelectedVertexIdx(null);
      setEditorNotice("Point removed from polygon.");
    },
    [imgDimensions, pushHistory]
  );

  // Delete whole selected coral mask
  const handleDeleteSelected = useCallback(() => {
    if (selectedId === null) return;
    const nextList = segments.filter((s) => s.id !== selectedId);
    setSegments(nextList);
    segmentsRef.current = nextList;
    pushHistory(nextList);
    setSelectedId(nextList.length > 0 ? nextList[0].id : null);
    setSelectedVertexIdx(null);
    setEditorNotice("Deleted selected coral annotation.");
  }, [selectedId, segments, pushHistory]);

  const handleDeleteCoral = useCallback(
    (id: number) => {
      const nextList = segments.filter((s) => s.id !== id);
      setSegments(nextList);
      segmentsRef.current = nextList;
      pushHistory(nextList);
      if (selectedId === id) {
        setSelectedId(nextList.length > 0 ? nextList[0].id : null);
        setSelectedVertexIdx(null);
      }
      setEditorNotice("Deleted coral annotation.");
    },
    [selectedId, segments, pushHistory]
  );

  // Update genus, growth form, condition of selected coral
  const handleUpdateSelectedMetadata = useCallback(
    (field: string, value: any) => {
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
    },
    [selectedId, segments, pushHistory]
  );

  // Toggle coral visibility
  const handleToggleVisibility = useCallback((id: number) => {
    setSegments((prev) =>
      prev.map((s) => (s.id === id ? { ...s, visible: !s.visible } : s))
    );
  }, []);

  return {
    segments,
    setSegments,
    segmentsRef,
    selectedId,
    setSelectedId,
    selectedVertexIdx,
    setSelectedVertexIdx,
    editorNotice,
    setEditorNotice,
    isSaving,
    saveStatus,
    hasUnsavedChanges,
    lastSavedTime,
    hasUnsavedRef,
    isSavingRef,
    loadPolygons,
    executeSave,
    addSegment,
    handleVertexDrag,
    handleVertexDragEnd,
    handleAddVertexOnEdge,
    handleDeleteVertex,
    handleDeleteSelected,
    handleDeleteCoral,
    handleUpdateSelectedMetadata,
    handleToggleVisibility,
  };
}
