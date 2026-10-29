import React, { useState } from "react";
import { Download, FileVideo, FileSpreadsheet, FileCode, Loader2 } from "lucide-react";
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
import { getVideoExportUrl } from "../../services/api";

interface VideoExportPanelProps {
  jobId: string;
  videoName: string;
  hasResults: boolean;
  rawJsonData: any;
}

export const VideoExportPanel: React.FC<VideoExportPanelProps> = ({
  jobId,
  videoName,
  hasResults,
  rawJsonData,
}) => {
  const [downloadingFormat, setDownloadingFormat] = useState<string | null>(null);
  const stem = videoName.replace(/\.[^/.]+$/, "");

  const handleDownload = (format: "mp4" | "csv" | "json") => {
    if (!jobId) return;
    setDownloadingFormat(format);
    const exportFmt = format === "json" ? "coco" : format;
    const url = getVideoExportUrl(jobId, exportFmt);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${stem}_${format === "json" ? "coco_video" : format === "csv" ? "transect" : "annotated"}.${format}`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      setDownloadingFormat(null);
    }, 1000);
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
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* 1. Annotated MP4 */}
          <Button
            variant="outline"
            disabled={!hasResults || !!downloadingFormat}
            onClick={() => handleDownload("mp4")}
            className="flex items-center justify-start gap-3 h-auto p-3 text-left border-border/70 hover:bg-muted/50 hover:border-border transition-all"
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
                Overlaid contours & telemetry
              </span>
            </div>
          </Button>

          {/* 2. Transect CSV */}
          <Button
            variant="outline"
            disabled={!hasResults || !!downloadingFormat}
            onClick={() => handleDownload("csv")}
            className="flex items-center justify-start gap-3 h-auto p-3 text-left border-border/70 hover:bg-muted/50 hover:border-border transition-all"
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
            className="flex items-center justify-start gap-3 h-auto p-3 text-left border-border/70 hover:bg-muted/50 hover:border-border transition-all"
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
