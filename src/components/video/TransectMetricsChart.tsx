import React, { useRef } from "react";
import { TransectMetricPoint } from "../../types";
import { Activity, Sparkles } from "lucide-react";

interface TransectMetricsChartProps {
  timeSeries: TransectMetricPoint[];
  currentTimeSec: number;
  durationSec: number;
  onSeek: (timeSec: number) => void;
}

export const TransectMetricsChart: React.FC<TransectMetricsChartProps> = ({
  timeSeries,
  currentTimeSec,
  durationSec,
  onSeek,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  if (!timeSeries || timeSeries.length === 0) {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 text-center text-slate-500 text-xs">
        Collecting survey transect telemetry as video plays...
      </div>
    );
  }

  // Calculate chart boundaries
  const maxCorals = Math.max(5, ...timeSeries.map((d) => d.active_corals_count));
  const maxTime = Math.max(durationSec || 1, timeSeries[timeSeries.length - 1]?.timestamp_sec || 1);

  // SVG dimensions
  const width = 800;
  const height = 110;
  const padding = { top: 12, right: 12, bottom: 20, left: 32 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Coordinate mappers
  const getX = (t: number) => padding.left + (t / maxTime) * plotWidth;
  const getYCorals = (c: number) =>
    padding.top + plotHeight - (c / maxCorals) * plotHeight;
  const getYBleach = (bPct: number) =>
    padding.top + plotHeight - (bPct / 100.0) * plotHeight;

  // Build SVG paths
  const coralPoints = timeSeries.map((d) => `${getX(d.timestamp_sec)},${getYCorals(d.active_corals_count)}`);
  const coralPath = `M ${coralPoints.join(" L ")}`;

  const bleachPoints = timeSeries.map((d) => `${getX(d.timestamp_sec)},${getYBleach(d.bleaching_prevalence_pct)}`);
  const bleachPath = `M ${bleachPoints.join(" L ")}`;

  // Current playhead X coordinate
  const currentX = getX(currentTimeSec);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!containerRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, (clickX - (padding.left / width) * rect.width) / ((plotWidth / width) * rect.width)));
    onSeek(ratio * maxTime);
  };

  return (
    <div ref={containerRef} className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 shadow-lg">
      <div className="flex items-center justify-between mb-1.5 text-xs">
        <div className="flex items-center gap-4">
          <span className="font-semibold text-slate-200 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            Transect Timeline Telemetry
          </span>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="flex items-center gap-1 text-cyan-400">
              <span className="w-2.5 h-0.5 bg-cyan-400 inline-block rounded" />
              Active Corals
            </span>
            <span className="flex items-center gap-1 text-rose-400">
              <span className="w-2.5 h-0.5 bg-rose-400 inline-block rounded" />
              Bleaching %
            </span>
            <span className="flex items-center gap-1 text-amber-400">
              <Sparkles className="w-2.5 h-2.5" />
              Keyframes
            </span>
          </div>
        </div>
        <span className="text-slate-500 text-[10px]">Click anywhere along chart to seek</span>
      </div>

      <div className="relative w-full overflow-hidden cursor-crosshair">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-24 select-none"
          onClick={handleClick}
        >
          {/* Grid lines */}
          <line
            x1={padding.left}
            y1={padding.top}
            x2={width - padding.right}
            y2={padding.top}
            stroke="#1e293b"
            strokeDasharray="3 3"
          />
          <line
            x1={padding.left}
            y1={padding.top + plotHeight / 2}
            x2={width - padding.right}
            y2={padding.top + plotHeight / 2}
            stroke="#1e293b"
            strokeDasharray="3 3"
          />
          <line
            x1={padding.left}
            y1={padding.top + plotHeight}
            x2={width - padding.right}
            y2={padding.top + plotHeight}
            stroke="#334155"
          />

          {/* Keyframe vertical pins */}
          {timeSeries
            .filter((d) => d.is_keyframe)
            .map((d) => (
              <line
                key={d.frame_index}
                x1={getX(d.timestamp_sec)}
                y1={padding.top}
                x2={getX(d.timestamp_sec)}
                y2={padding.top + plotHeight}
                stroke="#f59e0b"
                strokeWidth="1"
                opacity="0.4"
                strokeDasharray="2 2"
              />
            ))}

          {/* Coral Density Curve */}
          <path
            d={coralPath}
            fill="none"
            stroke="#06b6d4"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Bleaching Prevalence Curve */}
          <path
            d={bleachPath}
            fill="none"
            stroke="#f43f5e"
            strokeWidth="1.5"
            strokeDasharray="4 2"
            strokeLinecap="round"
          />

          {/* Active playhead vertical bar */}
          <line
            x1={currentX}
            y1={padding.top - 2}
            x2={currentX}
            y2={padding.top + plotHeight + 2}
            stroke="#ffffff"
            strokeWidth="1.5"
            opacity="0.9"
          />
          <circle cx={currentX} cy={padding.top + 2} r="3" fill="#ffffff" />

          {/* Y Axis Labels */}
          <text
            x={padding.left - 6}
            y={padding.top + 4}
            fill="#64748b"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
          >
            {maxCorals}
          </text>
          <text
            x={padding.left - 6}
            y={padding.top + plotHeight}
            fill="#64748b"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
          >
            0
          </text>
        </svg>
      </div>
    </div>
  );
};
