import { CoralSegment, SummaryStats, HealthSummary } from "../../types";

export type ToolMode = "select" | "sam" | "polygon";

export interface EditableSegment {
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

export interface SegmentationEditorModalProps {
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

export type SaveStatus = "saved" | "unsaved" | "saving";
