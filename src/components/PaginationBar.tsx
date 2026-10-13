import React, { useMemo } from "react";
import { ChevronLeft, ChevronRight, FileImage } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PaginationBarProps {
  currentIndex: number;
  totalImages: number;
  currentImageName: string;
  imageNames?: string[];
  itemStatuses?: Record<string, "pending" | "processing" | "completed" | "error" | "cancelled">;
  onPrev: () => void;
  onNext: () => void;
  onSelectPage: (index: number) => void;
}

/**
 * Truncates filename in the middle with ellipsis (e.g. imgdsas....jpg)
 * preserving the file extension and prefix when it exceeds maxLen.
 */
export function truncateMiddle(filename: string, maxLen: number = 32): string {
  if (!filename || filename.length <= maxLen) return filename;

  const extIdx = filename.lastIndexOf(".");
  const ext = extIdx > 0 ? filename.slice(extIdx) : "";
  const base = extIdx > 0 ? filename.slice(0, extIdx) : filename;

  const available = maxLen - ext.length - 3; // 3 for "..."
  if (available <= 6) {
    return `${base.slice(0, 6)}...${ext}`;
  }

  const front = Math.ceil(available / 2);
  const back = Math.floor(available / 2);

  return `${base.slice(0, front)}...${base.slice(-back)}${ext}`;
}

export const PaginationBar: React.FC<PaginationBarProps> = ({
  currentIndex,
  totalImages,
  currentImageName,
  imageNames = [],
  itemStatuses = {},
  onPrev,
  onNext,
  onSelectPage,
}) => {
  // Sliding window pagination:
  // If totalImages <= 7: [0, 1, 2, ..., totalImages - 1]
  // If totalImages > 7:
  //   Near start (currentIndex <= 3): [0, 1, 2, 3, 4, 'ellipsis-right', totalImages - 1]
  //   Near end (currentIndex >= totalImages - 4): [0, 'ellipsis-left', total - 5, total - 4, total - 3, total - 2, total - 1]
  //   Middle: [0, 'ellipsis-left', currentIndex - 1, currentIndex, currentIndex + 1, 'ellipsis-right', totalImages - 1]
  const visiblePages = useMemo<(number | "ellipsis-left" | "ellipsis-right")[]>(() => {
    if (totalImages <= 1) return [0];
    if (totalImages <= 7) {
      return Array.from({ length: totalImages }, (_, i) => i);
    }

    if (currentIndex <= 3) {
      return [0, 1, 2, 3, 4, "ellipsis-right", totalImages - 1];
    }

    if (currentIndex >= totalImages - 4) {
      return [
        0,
        "ellipsis-left",
        totalImages - 5,
        totalImages - 4,
        totalImages - 3,
        totalImages - 2,
        totalImages - 1,
      ];
    }

    return [
      0,
      "ellipsis-left",
      currentIndex - 1,
      currentIndex,
      currentIndex + 1,
      "ellipsis-right",
      totalImages - 1,
    ];
  }, [currentIndex, totalImages]);

  const renderPageButton = (p: number) => {
    const imgName = imageNames[p];
    const status = imgName ? itemStatuses[imgName] : undefined;
    const isCurrent = p === currentIndex;

    return (
      <Button
        key={p}
        variant={isCurrent ? "secondary" : "ghost"}
        size="sm"
        onClick={() => onSelectPage(p)}
        title={
          imgName
            ? `${imgName} (${status || "ready"})`
            : `Image ${p + 1}`
        }
        className={`relative h-8 min-w-8 px-2 font-mono text-xs font-semibold rounded-md transition-all ${
          isCurrent
            ? "bg-primary/10 text-primary border border-primary/30 shadow-2xs font-bold"
            : "text-muted-foreground hover:text-foreground hover:bg-muted"
        }`}
      >
        <span>{p + 1}</span>
        {status === "completed" && (
          <span
            className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-emerald-500 border border-background shadow-2xs"
            title="Processing completed"
          />
        )}
        {status === "processing" && (
          <span className="absolute -top-1 -right-1 flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-teal-500" />
          </span>
        )}
        {status === "pending" && (
          <span
            className="absolute -top-1 -right-1 h-1.5 w-1.5 rounded-full bg-slate-300 border border-background"
            title="Queued in background"
          />
        )}
        {status === "error" && (
          <span
            className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-red-500 border border-background shadow-2xs"
            title="Inference error"
          />
        )}
      </Button>
    );
  };

  return (
    <nav
      className="flex items-center justify-between w-full min-h-[48px] px-4 py-2 rounded-xl border border-border/80 bg-card shadow-xs select-none gap-4"
      aria-label="Gallery Navigation"
    >
      {/* Left: Image file badge with icon, truncated name, and index (never overlaps with buttons) */}
      <div className="flex items-center min-w-0">
        <div className="flex items-center gap-2.5 px-3.5 py-1.5 rounded-lg bg-muted/60 border border-border/60 shadow-2xs min-w-0">
          <FileImage className="size-3.5 text-primary shrink-0" />
          <span
            className="font-mono text-xs font-semibold text-foreground truncate max-w-[180px] sm:max-w-[280px] md:max-w-[340px]"
            title={currentImageName}
          >
            {truncateMiddle(currentImageName, 32)}
          </span>
          <span className="text-muted-foreground/40 shrink-0">•</span>
          <span className="text-xs text-muted-foreground font-medium shrink-0 whitespace-nowrap">
            <strong className="text-foreground font-bold">
              {currentIndex + 1}
            </strong>{" "}
            of {totalImages}
          </span>
        </div>
      </div>

      {/* Right: Consolidated Pagination Controls */}
      <div className="flex items-center gap-1.5 shrink-0">
        <Button
          variant="outline"
          size="sm"
          onClick={onPrev}
          disabled={currentIndex === 0}
          className="h-8 gap-1.5 px-2.5 sm:px-3 text-xs font-medium"
        >
          <ChevronLeft className="size-3.5" />
          <span className="hidden sm:inline">Previous</span>
        </Button>

        {visiblePages.map((p, idx) =>
          typeof p === "string" ? (
            <span
              key={`el-${idx}`}
              className="px-1 text-xs text-muted-foreground select-none font-mono"
            >
              •••
            </span>
          ) : (
            renderPageButton(p)
          )
        )}

        <Button
          variant="outline"
          size="sm"
          onClick={onNext}
          disabled={currentIndex >= totalImages - 1}
          className="h-8 gap-1.5 px-2.5 sm:px-3 text-xs font-medium"
        >
          <span className="hidden sm:inline">Next</span>
          <ChevronRight className="size-3.5" />
        </Button>
      </div>
    </nav>
  );
};
