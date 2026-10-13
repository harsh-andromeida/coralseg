import React, { useState, useEffect } from "react";
import {
  FolderArchive,
  FolderOpen,
  FolderPlus,
  Loader2,
  CheckCircle2,
  ExternalLink,
  AlertCircle,
  FileArchive,
  HardDrive,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  getDefaultExportPath,
  exportDatasetToFolder,
  pickFolderNative,
  openFolderNative,
  downloadBatchCocoZip,
} from "../services/api";
import { ExportFolderResponse } from "../types";

interface ExportDatasetModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  imageName?: string;
  totalImagesCount?: number;
}

export const ExportDatasetModal: React.FC<ExportDatasetModalProps> = ({
  open,
  onOpenChange,
  imageName,
  totalImagesCount,
}) => {
  const [folderPath, setFolderPath] = useState<string>("");
  const [includeZip, setIncludeZip] = useState<boolean>(false);
  const [openAfterExport, setOpenAfterExport] = useState<boolean>(true);
  const [isExportingFolder, setIsExportingFolder] = useState<boolean>(false);
  const [isExportingZip, setIsExportingZip] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<ExportFolderResponse | null>(null);
  const [zipResultPath, setZipResultPath] = useState<string | null>(null);

  // Initialize with system suggested default path when opened
  useEffect(() => {
    if (open) {
      setErrorMsg(null);
      setExportResult(null);
      setZipResultPath(null);
      getDefaultExportPath()
        .then((res) => {
          if (res?.default_folder) {
            setFolderPath(res.default_folder);
          }
        })
        .catch((err) => {
          console.warn("Could not retrieve default export path:", err);
        });
    }
  }, [open]);

  const handleBrowseFolder = async () => {
    try {
      const selected = await pickFolderNative(folderPath);
      if (selected) {
        setFolderPath(selected);
      }
    } catch (e: any) {
      console.warn("Folder picker error:", e);
    }
  };

  const handleExportToFolder = async () => {
    if (!folderPath.trim()) {
      setErrorMsg("Please specify a destination folder path.");
      return;
    }

    setIsExportingFolder(true);
    setErrorMsg(null);
    setExportResult(null);

    try {
      const res = await exportDatasetToFolder({
        folder_path: folderPath.trim(),
        include_zip: includeZip,
        open_folder: openAfterExport,
        image_names: imageName ? [imageName] : undefined,
      });

      setExportResult(res);
    } catch (e: any) {
      setErrorMsg(e?.message || "Failed to export dataset to folder.");
    } finally {
      setIsExportingFolder(false);
    }
  };

  const handleExportAsZip = async () => {
    setIsExportingZip(true);
    setErrorMsg(null);
    setExportResult(null);

    try {
      const res = await downloadBatchCocoZip(imageName);
      if (res.success && res.path) {
        setZipResultPath(res.path);
      } else if (res.error) {
        setErrorMsg(res.error);
      }
    } catch (e: any) {
      setErrorMsg(e?.message || "Failed to export COCO dataset zip.");
    } finally {
      setIsExportingZip(false);
    }
  };

  const handleOpenFolder = () => {
    if (exportResult?.folder_path) {
      openFolderNative(exportResult.folder_path);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl p-0 overflow-hidden border border-slate-200 shadow-2xl rounded-2xl bg-white">
        <DialogHeader className="px-6 pt-6 pb-4 bg-gradient-to-b from-slate-50 to-white border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="size-9 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700 shadow-xs">
              <FolderArchive className="size-5" />
            </div>
            <div>
              <DialogTitle className="text-base font-bold text-[#0f1e4a]">
                Export Dataset to Custom Folder
              </DialogTitle>
              <DialogDescription className="text-xs text-slate-500 mt-0.5">
                Export complete COCO annotations, full-res images, segments census CSV, and survey statistics{totalImagesCount ? ` across ${totalImagesCount} images` : ""}.
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="p-6 space-y-5">
          {errorMsg && (
            <div className="flex items-start gap-2.5 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs">
              <AlertCircle className="size-4 shrink-0 mt-0.5 text-rose-600" />
              <div className="flex-1 font-medium">{errorMsg}</div>
            </div>
          )}

          {exportResult && (
            <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900 space-y-2">
              <div className="flex items-center gap-2 text-xs font-bold text-emerald-800">
                <CheckCircle2 className="size-4 text-emerald-600" />
                <span>Dataset exported successfully to custom folder!</span>
              </div>
              <p className="text-[11px] text-emerald-700 font-mono break-all bg-emerald-100/70 p-2 rounded-lg border border-emerald-200/80">
                {exportResult.folder_path}
              </p>
              <div className="flex items-center justify-between pt-1 text-xs">
                <span className="text-emerald-700 font-medium">
                  {exportResult.total_images} images • {exportResult.total_corals} corals detected • {exportResult.written_files_count} files created
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={handleOpenFolder}
                  className="h-7 text-xs gap-1 border-emerald-300 text-emerald-800 hover:bg-emerald-100"
                >
                  <ExternalLink size={12} />
                  <span>Open Folder</span>
                </Button>
              </div>
            </div>
          )}

          {zipResultPath && (
            <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900 space-y-1">
              <div className="flex items-center gap-2 text-xs font-bold text-emerald-800">
                <CheckCircle2 className="size-4 text-emerald-600" />
                <span>COCO Dataset (.zip) saved successfully!</span>
              </div>
              <p className="text-[11px] text-emerald-700 font-mono break-all">
                {zipResultPath}
              </p>
            </div>
          )}

          {/* Destination Folder Selector */}
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-800 flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <HardDrive className="size-3.5 text-slate-600" />
                Destination Folder Path
              </span>
              <span className="text-[11px] font-normal text-slate-500">
                Target directory for dataset
              </span>
            </label>

            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <input
                  type="text"
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder="/Users/username/Desktop/reefvision_dataset"
                  className="w-full h-10 px-3 text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                />
              </div>
              <Button
                type="button"
                variant="outline"
                onClick={handleBrowseFolder}
                className="h-10 px-3.5 text-xs font-semibold gap-1.5 border-slate-200 bg-white hover:bg-slate-50 rounded-xl shrink-0 text-slate-700 shadow-2xs"
                title="Browse and select a custom folder"
              >
                <FolderOpen className="size-4 text-primary" />
                <span>Browse...</span>
              </Button>
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              Files will be written to this folder: <code className="text-slate-700 bg-slate-100 px-1 py-0.5 rounded">annotations/instances_default.json</code>, <code className="text-slate-700 bg-slate-100 px-1 py-0.5 rounded">images/</code>, <code className="text-slate-700 bg-slate-100 px-1 py-0.5 rounded">summary.csv</code>, and <code className="text-slate-700 bg-slate-100 px-1 py-0.5 rounded">README.md</code>.
            </p>
          </div>

          {/* Export Options */}
          <div className="p-3.5 bg-slate-50/70 border border-slate-150 rounded-xl space-y-2.5">
            <label className="flex items-center gap-2 cursor-pointer select-none text-xs text-slate-700 font-medium">
              <input
                type="checkbox"
                checked={includeZip}
                onChange={(e) => setIncludeZip(e.target.checked)}
                className="rounded border-slate-300 text-primary focus:ring-primary/30 size-3.5"
              />
              <span>Also generate a standalone bundled <strong>.zip archive</strong> inside the destination folder</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer select-none text-xs text-slate-700 font-medium">
              <input
                type="checkbox"
                checked={openAfterExport}
                onChange={(e) => setOpenAfterExport(e.target.checked)}
                className="rounded border-slate-300 text-primary focus:ring-primary/30 size-3.5"
              />
              <span>Reveal destination folder in Finder / File Explorer upon completion</span>
            </label>
          </div>
        </div>

        <DialogFooter className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-3">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={isExportingFolder || isExportingZip}
            onClick={handleExportAsZip}
            className="text-xs text-slate-600 hover:text-slate-900 gap-1.5 h-9"
            title="Download or save as a single compressed .zip file"
          >
            {isExportingZip ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <FileArchive className="size-3.5" />
            )}
            <span>Save as .ZIP Archive Instead</span>
          </Button>

          <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={isExportingFolder || isExportingZip}
              onClick={() => onOpenChange(false)}
              className="text-xs h-9 px-4 rounded-xl border-slate-200"
            >
              Close
            </Button>

            <Button
              type="button"
              variant="default"
              size="sm"
              disabled={isExportingFolder || isExportingZip || !folderPath.trim()}
              onClick={handleExportToFolder}
              className="text-xs font-bold h-9 px-4 rounded-xl gap-2 text-white bg-gradient-to-r from-[#0f1e4a] via-[#163e80] to-[#0d7c85] hover:opacity-95 shadow-xs transition-all"
            >
              {isExportingFolder ? (
                <>
                  <Loader2 className="size-3.5 animate-spin" />
                  <span>Exporting to Folder...</span>
                </>
              ) : (
                <>
                  <FolderPlus className="size-4" />
                  <span>Export to Folder</span>
                </>
              )}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
