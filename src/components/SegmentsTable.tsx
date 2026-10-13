import React, { useState, useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  SortingState,
} from "@tanstack/react-table";
import { ArrowUpDown, ArrowUp, ArrowDown, Layers, Search, Edit3 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { InfoTooltip } from "./InfoTooltip";
import { CoralSegment, SummaryStats } from "../types";

interface SegmentsTableProps {
  segments: CoralSegment[];
  stats: SummaryStats;
  selectedSegmentId?: number | null;
  onSelectSegment?: (id: number | null) => void;
  isLoading?: boolean;
  onOpenEditor?: () => void;
  onEditSegment?: (segmentId: number) => void;
}

const columnHelper = createColumnHelper<CoralSegment>();

const ConfBar = ({ val, color = "bg-primary" }: { val: number; color?: string }) => (
  <div className="flex items-center gap-2 min-w-[100px]">
    <Progress value={val} className="h-1.5 w-16 bg-muted" indicatorColor={color} />
    <span className="text-[11px] font-mono text-muted-foreground font-medium">
      {val.toFixed(1)}%
    </span>
  </div>
);

export const SegmentsTable: React.FC<SegmentsTableProps> = ({
  segments,
  stats,
  selectedSegmentId,
  onSelectSegment,
  isLoading = false,
  onOpenEditor,
  onEditSegment,
}) => {
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(
    () => [
      columnHelper.accessor("id", {
        header: "ID",
        cell: (info) => {
          const s = info.row.original;
          return (
            <div className="flex items-center gap-2 font-mono text-xs font-semibold">
              <span
                className="size-2.5 rounded-full ring-1 ring-black/10 shrink-0"
                style={{ backgroundColor: s.color_hex }}
              />
              <span>{s.id_str}</span>
            </div>
          );
        },
      }),
      columnHelper.accessor("genus", {
        header: "Taxon Genus",
        cell: (info) => (
          <span className="font-semibold text-foreground">{info.getValue()}</span>
        ),
      }),
      columnHelper.accessor("growth_form", {
        header: "Growth Form",
        cell: (info) => (
          <span className="text-muted-foreground font-medium">{info.getValue()}</span>
        ),
      }),
      columnHelper.accessor("taxon_conf", {
        header: "Taxon Conf",
        cell: (info) => <ConfBar val={info.getValue()} color="bg-primary" />,
      }),
      columnHelper.accessor("condition", {
        header: "Condition",
        cell: (info) => {
          const cond = info.getValue();
          const isBleached = cond === "Bleached";
          return (
            <Badge
              variant={isBleached ? "destructive" : "success"}
              className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5"
            >
              {cond}
            </Badge>
          );
        },
      }),
      columnHelper.accessor("condition_conf", {
        header: "Condition Conf",
        cell: (info) => (
          <ConfBar
            val={info.getValue()}
            color={
              info.row.original.condition === "Bleached"
                ? "bg-rose-500"
                : "bg-emerald-500"
            }
          />
        ),
      }),
      columnHelper.accessor("area_pct", {
        header: "Area (%)",
        cell: (info) => (
          <span className="font-mono text-xs font-medium text-foreground">
            {info.getValue()}%
          </span>
        ),
      }),
      columnHelper.accessor("area_px", {
        header: "Area (px)",
        cell: (info) => (
          <span className="font-mono text-xs text-muted-foreground">
            {info.getValue().toLocaleString("en-US")}
          </span>
        ),
      }),
      columnHelper.accessor("predicted_iou", {
        header: "IoU Confidence",
        cell: (info) => (
          <span className="font-mono text-xs font-semibold text-foreground">
            {info.row.original.is_human_annotated ? "1.000 (Human)" : info.getValue().toFixed(3)}
          </span>
        ),
      }),
      columnHelper.display({
        id: "actions",
        header: "Action",
        cell: (info) => (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              if (onEditSegment) {
                onEditSegment(info.row.original.id);
              } else if (onOpenEditor) {
                onOpenEditor();
              }
            }}
            className="h-6 px-2 text-[10px] font-semibold text-primary hover:text-primary hover:bg-primary/10 gap-1 cursor-pointer"
            title="Edit this coral mask in Annotation Studio"
          >
            <Edit3 className="size-3" />
            <span>Edit</span>
          </Button>
        ),
      }),
    ],
    [onEditSegment, onOpenEditor]
  );

  const table = useReactTable({
    data: segments,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <Card className="border-border/80 bg-card shadow-sm rounded-xl overflow-hidden">
      <CardHeader className="py-3 px-6 flex flex-row items-center justify-between gap-4 border-b border-border/70 bg-card">
        <div className="flex items-center gap-2 min-w-0">
          <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2 shrink-0">
            <Layers className="size-4 text-primary" />
            <span>Detected Coral Segments Breakdown</span>
          </CardTitle>
          <InfoTooltip content="Dense instance boundaries, taxonomy identification, and benthic condition assessment" />
        </div>

        <div className="flex items-center gap-2 shrink-0 flex-nowrap">
          {onOpenEditor && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onOpenEditor}
              className="h-7 px-2.5 text-xs font-semibold gap-1.5 text-primary border-primary/25 bg-primary/10 hover:bg-primary/20 cursor-pointer mr-1"
              title="Open Manual Segmentation Studio"
            >
              <Edit3 className="size-3.5" />
              <span>Edit Segmentation</span>
            </Button>
          )}
          <Badge
            variant="outline"
            className="font-mono text-xs bg-muted/40 text-muted-foreground border-border/80 shrink-0"
          >
            <span className="text-muted-foreground/60 mr-1 font-sans">Resolution:</span>
            {stats.image_resolution && stats.image_resolution !== "0x0"
              ? stats.image_resolution.replace("x", " × ")
              : "2048 × 1024"}
          </Badge>
          <Badge
            variant="outline"
            className="font-mono text-xs bg-muted/40 text-muted-foreground border-border/80 shrink-0"
          >
            <span className="text-muted-foreground/60 mr-1 font-sans">Coverage:</span>
            {isLoading ? (
              <Skeleton className="h-3.5 w-16 inline-block align-middle ml-1" />
            ) : (
              <>
                <span className="text-primary font-semibold">{stats.coral_coverage_pct}%</span>
                <span className="text-muted-foreground/60 ml-1">
                  ({stats.coral_covered_pixels.toLocaleString("en-US")} px)
                </span>
              </>
            )}
          </Badge>
          <Badge variant="coral" className="text-xs font-medium px-2.5 py-0.5 shrink-0">
            {isLoading ? (
              <Skeleton className="h-3.5 w-14 inline-block align-middle" />
            ) : (
              `${stats.total_corals_detected} Colonies`
            )}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {isLoading ? (
          <div className="relative w-full overflow-auto max-h-[460px]">
            <Table>
              <TableHeader className="sticky top-0 bg-muted/80 backdrop-blur-xs z-10 border-b border-border">
                <TableRow>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    ID
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Taxon Genus
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Growth Form
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Taxon Conf
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Condition
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Condition Conf
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Area (%)
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Area (px)
                  </TableHead>
                  <TableHead className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    IoU Confidence
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Array.from({ length: 6 }).map((_, idx) => (
                  <TableRow key={idx} className="hover:bg-transparent">
                    {/* ID */}
                    <TableCell className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Skeleton className="size-2.5 rounded-full shrink-0" />
                        <Skeleton className="h-3.5 w-12" />
                      </div>
                    </TableCell>
                    {/* Taxon Genus */}
                    <TableCell className="px-4 py-3">
                      <Skeleton
                        className="h-3.5"
                        style={{ width: `${65 + (idx % 4) * 16}px` }}
                      />
                    </TableCell>
                    {/* Growth Form */}
                    <TableCell className="px-4 py-3">
                      <Skeleton
                        className="h-3.5"
                        style={{ width: `${55 + ((idx + 1) % 4) * 18}px` }}
                      />
                    </TableCell>
                    {/* Taxon Conf */}
                    <TableCell className="px-4 py-3">
                      <div className="flex items-center gap-2 min-w-[100px]">
                        <Skeleton className="h-1.5 w-16" />
                        <Skeleton className="h-3 w-8" />
                      </div>
                    </TableCell>
                    {/* Condition */}
                    <TableCell className="px-4 py-3">
                      <Skeleton className="h-5 w-16 rounded-full" />
                    </TableCell>
                    {/* Condition Conf */}
                    <TableCell className="px-4 py-3">
                      <div className="flex items-center gap-2 min-w-[100px]">
                        <Skeleton className="h-1.5 w-16" />
                        <Skeleton className="h-3 w-8" />
                      </div>
                    </TableCell>
                    {/* Area (%) */}
                    <TableCell className="px-4 py-3">
                      <Skeleton className="h-3.5 w-10" />
                    </TableCell>
                    {/* Area (px) */}
                    <TableCell className="px-4 py-3">
                      <Skeleton className="h-3.5 w-14" />
                    </TableCell>
                    {/* IoU Confidence */}
                    <TableCell className="px-4 py-3">
                      <Skeleton className="h-3.5 w-12" />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : segments.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
            <Search className="size-8 mb-3 opacity-30 text-muted-foreground" />
            <div className="text-sm font-semibold text-foreground">
              No Coral Colonies Detected
            </div>
            <p className="text-xs text-muted-foreground mt-1 max-w-sm">
              Adjust minimum mask area or IoU threshold in Analysis Controls to detect smaller colonies.
            </p>
          </div>
        ) : (
          <div className="relative w-full overflow-auto max-h-[460px]">
            <Table>
              <TableHeader className="sticky top-0 bg-muted/80 backdrop-blur-xs z-10 border-b border-border">
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => {
                      const sortDir = header.column.getIsSorted();
                      return (
                        <TableHead
                          key={header.id}
                          onClick={header.column.getToggleSortingHandler()}
                          className="h-10 px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground cursor-pointer select-none hover:bg-muted transition-colors"
                        >
                          <div className="flex items-center gap-1.5">
                            {flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                            <span className="shrink-0 text-muted-foreground/60">
                              {sortDir === "asc" ? (
                                <ArrowUp className="size-3 text-primary" />
                              ) : sortDir === "desc" ? (
                                <ArrowDown className="size-3 text-primary" />
                              ) : (
                                <ArrowUpDown className="size-3 opacity-30" />
                              )}
                            </span>
                          </div>
                        </TableHead>
                      );
                    })}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows.map((row) => {
                  const isSelected = selectedSegmentId === row.original.id;
                  return (
                    <TableRow
                      key={row.id}
                      data-state={isSelected ? "selected" : undefined}
                      onClick={() =>
                        onSelectSegment &&
                        onSelectSegment(isSelected ? null : row.original.id)
                      }
                      className="cursor-pointer hover:bg-primary/5 transition-colors data-[state=selected]:bg-primary/10"
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell key={cell.id} className="px-4 py-3">
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext()
                          )}
                        </TableCell>
                      ))}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
