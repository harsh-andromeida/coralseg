import React, { useState } from "react";
import { Download, FileVideo, FileSpreadsheet, FileCode, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { downloadVideoExport } from "../../services/api";

interface VideoExportPanelProps {
  jobId: string;
  videoName: string;
  hasResults: boolean;
  rawJsonData: any;
  exportPaths?: Record<string, string | null>;
  isProcessing?: boolean;
}

export const VideoExportPanel: React.FC<VideoExportPanelProps> = ({
  jobId,
  videoName,
  hasResults,
  rawJsonData,
  exportPaths = {},
  isProcessing = false,
}) => {
  const [downloadingFormat, setDownloadingFormat] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const stem = videoName.replace(/\.[^/.]+$/, "");

  const handleDownload = async (format: "mp4" | "csv" | "json") => {
    if (!jobId) return;
    setDownloadingFormat(format);
    setFeedback(null);

    const exportFmt = format === "json" ? "coco" : format;
    const defaultName = `${stem}_${format === "json" ? "coco_video" : format === "csv" ? "transect" : "annotated"}.${format}`;
    const sourceKey = format === "json" ? "coco_json" : format === "csv" ? "transect_csv" : "annotated_mp4";
    const sourcePath = exportPaths[sourceKey];

    try {
      const res = await downloadVideoExport(jobId, exportFmt, defaultName, sourcePath);
      if (res.cancelled) {
        setDownloadingFormat(null);
        return;
      }
      if (res.success) {
        setFeedback({
          type: "success",
          message: `Saved ${defaultName} successfully!`,
        });
        setTimeout(() => setFeedback(null), 4000);
      } else if (res.error) {
        setFeedback({
          type: "error",
          message: res.error,
        });
      }
    } catch (err: any) {
      setFeedback({
        type: "error",
        message: err?.message || "Failed to download export file.",
      });
    } finally {
      setDownloadingFormat(null);
    }
  };

  return (
    <Card className="border-border/80 bg-card shadow-sm rounded-xl overflow-hidden">
      <CardHeader className="py-4 px-6 border-b border-border/70 bg-card">
        <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Download className="size-4 text-primary" />
          <span>Export & Transect Data Inspector</span>
        </CardTitle>
        <CardDescription className="text-xs text-muted-foreground mt-0.5">
          Export full transect telemetry time-series, annotated video with overlaid SAM 2 masks, and MOT/COCO-Video datasets
        </CardDescription>
      </CardHeader>

      <CardContent className="p-6 flex flex-col gap-4">
        {feedback && (
          <div
            className={`p-3 rounded-lg flex items-center gap-2 text-xs font-medium ${
              feedback.type === "success"
                ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"
                : "bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20"
            }`}
          >
            {feedback.type === "success" ? (
              <CheckCircle2 className="size-4 shrink-0" />
            ) : (
              <AlertCircle className="size-4 shrink-0" />
            )}
            <span>{feedback.message}</span>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* 1. Annotated MP4 */}
          <Button
            variant="outline"
            disabled={!hasResults || !!downloadingFormat}
            onClick={() => handleDownload("mp4")}
            className="flex items-center justify-start gap-3 h-auto p-3 text-left border-border/70 hover:bg-muted/50 hover:border-border transition-all cursor-pointer"
          >
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-500 shrink-0">
              {downloadingFormat === "mp4" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileVideo className="size-4" />
              )}
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-semibold text-foreground">Annotated Video (.mp4)</span>
              <span className="text-[11px] text-muted-foreground truncate">
                {isProcessing ? "Compile processed frames" : "Overlaid contours & telemetry"}
              </span>
            </div>
          </Button>

          {/* 2. Transect CSV */}
          <Button
            variant="outline"
            disabled={!hasResults || !!downloadingFormat}
            onClick={() => handleDownload("csv")}
            className="flex items-center justify-start gap-3 h-auto p-3 text-left border-border/70 hover:bg-muted/50 hover:border-border transition-all cursor-pointer"
          >
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-500 shrink-0">
              {downloadingFormat === "csv" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileSpreadsheet className="size-4" />
              )}
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-semibold text-foreground">Transect Telemetry (.csv)</span>
              <span className="text-[11px] text-muted-foreground truncate">
                Frame-by-frame scientific log
              </span>
            </div>
          </Button>

          {/* 3. COCO Video JSON */}
          <Button
            variant="outline"
            disabled={!hasResults || !!downloadingFormat}
            onClick={() => handleDownload("json")}
            className="flex items-center justify-start gap-3 h-auto p-3 text-left border-border/70 hover:bg-muted/50 hover:border-border transition-all cursor-pointer"
          >
            <div className="p-2 rounded-lg bg-primary/10 text-primary shrink-0">
              {downloadingFormat === "json" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileCode className="size-4" />
              )}
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-semibold text-foreground">COCO-Video Tracking (.json)</span>
              <span className="text-[11px] text-muted-foreground truncate">
                MOT tracking format
              </span>
            </div>
          </Button>
        </div>

        {/* JSON Telemetry Accordion Inspector */}
        <Accordion type="single" collapsible className="w-full border-t border-border/70 pt-2">
          <AccordionItem value="json-inspector" className="border-none">
            <AccordionTrigger className="py-2 text-xs font-medium text-muted-foreground hover:text-foreground">
              <span>View Raw Frame Telemetry & Tracking Detections (JSON)</span>
            </AccordionTrigger>
            <AccordionContent>
              <pre className="p-3 bg-muted/40 rounded-lg text-[11px] font-mono text-muted-foreground overflow-x-auto max-h-60 border border-border/60">
                {JSON.stringify(rawJsonData, null, 2)}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
};
