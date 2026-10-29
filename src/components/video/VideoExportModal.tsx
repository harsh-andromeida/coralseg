import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { VideoJobStatus } from "../../types";
import { getVideoExportUrl } from "../../services/api";
import { Download, FileVideo, FileSpreadsheet, Code2, Check, AlertCircle } from "lucide-react";

interface VideoExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  jobStatus: VideoJobStatus | null;
}

export const VideoExportModal: React.FC<VideoExportModalProps> = ({
  isOpen,
  onClose,
  jobStatus,
}) => {
  const [downloadingFormat, setDownloadingFormat] = useState<string | null>(null);

  if (!jobStatus) return null;

  const handleDownload = (format: "mp4" | "csv" | "coco") => {
    setDownloadingFormat(format);
    const url = getVideoExportUrl(jobStatus.job_id, format);

    const a = document.createElement("a");
    a.href = url;
    a.download = `${jobStatus.job_id}_${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    setTimeout(() => {
      setDownloadingFormat(null);
    }, 1500);
  };

  const isCompleted = jobStatus.state === "completed";

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-xl bg-slate-900 border-slate-800 text-slate-100">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold flex items-center gap-2 text-cyan-400">
            <Download className="w-5 h-5" />
            Export Transect Tracking Results
          </DialogTitle>
          <DialogDescription className="text-slate-400 text-xs">
            Download processed survey video, scientific time-series spreadsheets, or computer vision training datasets.
          </DialogDescription>
        </DialogHeader>

        {!isCompleted && (
          <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg flex items-center gap-2.5 text-xs text-amber-300">
            <AlertCircle className="w-4 h-4 shrink-0 text-amber-400" />
            <span>
              Video processing is currently in progress ({jobStatus.progress_pct}%). Full exports will be compiled upon completion.
            </span>
          </div>
        )}

        <div className="grid grid-cols-1 gap-3 py-2">
          {/* Annotated MP4 */}
          <div className="flex items-center justify-between p-3.5 bg-slate-800/60 border border-slate-700/60 rounded-xl hover:border-slate-600 transition-colors">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                <FileVideo className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-200">Annotated MP4 Video</h4>
                <p className="text-xs text-slate-400">
                  Full-length video with rendered coral contours, persistent track IDs, and HUD telemetry banner.
                </p>
              </div>
            </div>
            <Button
              size="sm"
              disabled={!isCompleted || downloadingFormat === "mp4"}
              onClick={() => handleDownload("mp4")}
              className="bg-cyan-500 text-slate-950 hover:bg-cyan-400 font-semibold shrink-0"
            >
              {downloadingFormat === "mp4" ? (
                <>
                  <Check className="w-3.5 h-3.5 mr-1" /> Ready
                </>
              ) : (
                <>
                  <Download className="w-3.5 h-3.5 mr-1" /> Download MP4
                </>
              )}
            </Button>
          </div>

          {/* Scientific Transect CSV */}
          <div className="flex items-center justify-between p-3.5 bg-slate-800/60 border border-slate-700/60 rounded-xl hover:border-slate-600 transition-colors">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <FileSpreadsheet className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-200">Scientific Transect CSV</h4>
                <p className="text-xs text-slate-400">
                  Frame-by-frame log of active coral counts, cumulative colonies, bleaching %, and camera translation.
                </p>
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              disabled={downloadingFormat === "csv" || jobStatus.time_series?.length === 0}
              onClick={() => handleDownload("csv")}
              className="border-slate-700 hover:bg-slate-800 text-slate-200 shrink-0"
            >
              {downloadingFormat === "csv" ? (
                <>
                  <Check className="w-3.5 h-3.5 mr-1" /> Ready
                </>
              ) : (
                <>
                  <Download className="w-3.5 h-3.5 mr-1" /> Download CSV
                </>
              )}
            </Button>
          </div>

          {/* COCO-Video / MOT JSON */}
          <div className="flex items-center justify-between p-3.5 bg-slate-800/60 border border-slate-700/60 rounded-xl hover:border-slate-600 transition-colors">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                <Code2 className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-200">MOT / COCO-Video Annotations</h4>
                <p className="text-xs text-slate-400">
                  Standard multi-object tracking dataset annotations compatible with Roboflow, CVAT, and PyTorch.
                </p>
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              disabled={!isCompleted || downloadingFormat === "coco"}
              onClick={() => handleDownload("coco")}
              className="border-slate-700 hover:bg-slate-800 text-slate-200 shrink-0"
            >
              {downloadingFormat === "coco" ? (
                <>
                  <Check className="w-3.5 h-3.5 mr-1" /> Ready
                </>
              ) : (
                <>
                  <Download className="w-3.5 h-3.5 mr-1" /> Download JSON
                </>
              )}
            </Button>
          </div>
        </div>

        <DialogFooter className="mt-2">
          <Button variant="ghost" onClick={onClose} className="text-slate-400 hover:text-white">
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
