import React from "react";
import { Trash2, DownloadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { ModelSpec } from "../../types";
import { MODEL_GROUPS, isGroupReady } from "./modelGroups";

export interface ModelManagementModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  allDownloaded: boolean;
  modelsList: ModelSpec[];
  isDeletingModel: string | null;
  isDownloading: boolean;
  onDeleteModel: (groupId: string) => void;
  onStartDownload: (groupId: string) => void;
}

export const ModelManagementModal: React.FC<ModelManagementModalProps> = ({
  open,
  onOpenChange,
  allDownloaded,
  modelsList,
  isDeletingModel,
  isDownloading,
  onDeleteModel,
  onStartDownload,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md p-6">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <Trash2 className="h-5 w-5 text-red-500" />
            <DialogTitle className="text-base font-bold text-[#0f1e4a]">
              Manage Foundation Models
            </DialogTitle>
          </div>
          <DialogDescription className="text-xs text-slate-500 pt-1">
            Delete individual models to free disk space. You can re-download any deleted model at any time.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2 my-2 max-h-60 overflow-y-auto">
          {MODEL_GROUPS.map((group) => {
            const ready = isGroupReady(group.id, allDownloaded, modelsList);
            return (
              <div
                key={group.id}
                className="flex items-center justify-between p-2.5 rounded-lg border border-slate-100 bg-slate-50/60"
              >
                <div>
                  <div className="text-xs font-bold text-slate-800">{group.name}</div>
                  <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
                    <span>{group.task}</span>
                    <span>•</span>
                    <span>{group.size}</span>
                  </div>
                </div>
                <div>
                  {ready ? (
                    <Button
                      type="button"
                      variant="destructive"
                      size="sm"
                      className="h-7 px-2.5 text-xs gap-1.5 rounded-md"
                      onClick={() => onDeleteModel(group.id)}
                      disabled={!!isDeletingModel}
                      title={`Delete ${group.name}`}
                    >
                      <Trash2
                        className={`h-3.5 w-3.5 ${
                          isDeletingModel === group.id ? "animate-spin" : ""
                        }`}
                      />
                      <span>{isDeletingModel === group.id ? "Deleting..." : "Delete"}</span>
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 px-2.5 text-xs gap-1.5 rounded-md border-teal-200 text-teal-700 hover:bg-teal-50"
                      onClick={() => {
                        onStartDownload(group.id);
                        onOpenChange(false);
                      }}
                      disabled={isDownloading || !!isDeletingModel}
                      title={`Download ${group.name}`}
                    >
                      <DownloadCloud className="h-3.5 w-3.5" />
                      <span>Download</span>
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <DialogFooter className="mt-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onOpenChange(false)}
            disabled={!!isDeletingModel}
          >
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
