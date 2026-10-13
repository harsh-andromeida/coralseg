import React from "react";
import { Plus, Eye, EyeOff, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EditableSegment } from "./types";
import { TAXONOMY_OPTIONS, GROWTH_FORMS } from "./constants";

interface EditorSidebarProps {
  segments: EditableSegment[];
  selectedId: number | null;
  formGenus: string;
  formGrowthForm: string;
  formCondition: "Healthy" | "Bleached";
  onSelectCoral: (id: number) => void;
  onDeselectCoral: () => void;
  onDeleteSelected: () => void;
  onDeleteCoral: (id: number) => void;
  onToggleVisibility: (id: number) => void;
  onAddCoralClick: () => void;
  onUpdateMetadata: (field: string, value: any) => void;
  onSetFormGenus: (val: string) => void;
  onSetFormGrowthForm: (val: string) => void;
  onSetFormCondition: (val: "Healthy" | "Bleached") => void;
}

export const EditorSidebar: React.FC<EditorSidebarProps> = ({
  segments,
  selectedId,
  formGenus,
  formGrowthForm,
  formCondition,
  onSelectCoral,
  onDeselectCoral,
  onDeleteSelected,
  onDeleteCoral,
  onToggleVisibility,
  onAddCoralClick,
  onUpdateMetadata,
  onSetFormGenus,
  onSetFormGrowthForm,
  onSetFormCondition,
}) => {
  const activeSegment = segments.find((s) => s.id === selectedId) || null;

  return (
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
                  onUpdateMetadata("genus", val);
                  if (matched) {
                    onUpdateMetadata("growth_form", matched.form);
                  }
                } else {
                  onSetFormGenus(val);
                  if (matched) {
                    onSetFormGrowthForm(matched.form);
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
                  onUpdateMetadata("growth_form", val);
                } else {
                  onSetFormGrowthForm(val);
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
                    onUpdateMetadata("condition", "Healthy");
                  } else {
                    onSetFormCondition("Healthy");
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
                    onUpdateMetadata("condition", "Bleached");
                  } else {
                    onSetFormCondition("Bleached");
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
                onClick={onDeleteSelected}
                className="flex-1 h-7 text-xs text-red-600 border-red-200 bg-red-50 hover:bg-red-100 hover:text-red-700 cursor-pointer"
              >
                <Trash2 size={12} className="mr-1" />
                <span>Delete Mask</span>
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onDeselectCoral}
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
            onClick={onAddCoralClick}
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
                  onClick={() => onSelectCoral(seg.id)}
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
                        onToggleVisibility(seg.id);
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
                        onDeleteCoral(seg.id);
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
  );
};
