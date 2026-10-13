import { EditableSegment } from "./types";

// Deep clone segments to ensure immutable history snapshots
export const cloneSegments = (items: EditableSegment[]): EditableSegment[] => {
  return items.map((s) => ({
    ...s,
    polygon: s.polygon.map(([x, y]) => [x, y] as [number, number]),
  }));
};

// Extract unique vertices (omitting duplicate closing loop vertex if present)
export const getUniquePolygonPoints = (poly: [number, number][]): [number, number][] => {
  if (!poly || poly.length === 0) return [];
  if (
    poly.length >= 2 &&
    poly[0][0] === poly[poly.length - 1][0] &&
    poly[0][1] === poly[poly.length - 1][1]
  ) {
    return poly.slice(0, -1);
  }
  return poly;
};

// Ensure polygon ends with the first point to close the ring
export const ensureClosedPolygon = (pts: [number, number][]): [number, number][] => {
  if (!pts || pts.length < 3) return pts;
  if (
    pts[0][0] !== pts[pts.length - 1][0] ||
    pts[0][1] !== pts[pts.length - 1][1]
  ) {
    return [...pts, pts[0]];
  }
  return pts;
};

// Find the closest edge in polygon to a given point and compute perpendicular projection
export const findClosestEdge = (
  point: [number, number],
  vertices: [number, number][]
): { edgeIdx: number; projected: [number, number]; distance: number } | null => {
  if (!vertices || vertices.length < 2) return null;
  let minDistance = Infinity;
  let bestEdgeIdx = 0;
  let bestProjected: [number, number] = vertices[0];

  for (let i = 0; i < vertices.length; i++) {
    const p1 = vertices[i];
    const p2 = vertices[(i + 1) % vertices.length];
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    const lenSq = dx * dx + dy * dy;
    let t = 0;
    if (lenSq > 0) {
      t = Math.max(0, Math.min(1, ((point[0] - p1[0]) * dx + (point[1] - p1[1]) * dy) / lenSq));
    }
    const projX = p1[0] + t * dx;
    const projY = p1[1] + t * dy;
    const dist = Math.hypot(point[0] - projX, point[1] - projY);
    if (dist < minDistance) {
      minDistance = dist;
      bestEdgeIdx = i;
      bestProjected = [Math.round(projX), Math.round(projY)];
    }
  }
  return { edgeIdx: bestEdgeIdx, projected: bestProjected, distance: minDistance };
};

// Calculate polygon area (Shoelace formula) and reef coverage percentage
export const calculatePolygonArea = (
  poly: [number, number][],
  imgWidth: number,
  imgHeight: number
): { area_px: number; area_pct: number } => {
  if (!poly || poly.length < 3) return { area_px: 0, area_pct: 0 };
  let sum = 0;
  for (let i = 0; i < poly.length; i++) {
    const j = (i + 1) % poly.length;
    sum += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
  }
  const area_px = Math.round(Math.abs(sum) / 2);
  const totalPixels = Math.max(imgWidth * imgHeight, 1);
  const area_pct = Number(((area_px / totalPixels) * 100).toFixed(2));
  return { area_px, area_pct };
};
