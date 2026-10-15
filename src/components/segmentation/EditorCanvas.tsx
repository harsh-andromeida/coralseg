import React, { useState, useEffect, useRef } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImage,
  Line,
  Circle,
  Group,
  Text as KonvaText,
  Rect,
} from "react-konva";
import Konva from "konva";
import { EditableSegment, ToolMode } from "./types";
import { PromptPoint } from "../../types";
import { getUniquePolygonPoints, findClosestEdge } from "./geometry";

interface EditorCanvasProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  stageRef: React.RefObject<Konva.Stage | null>;
  htmlImage: HTMLImageElement | null;
  imgDimensions: { w: number; h: number };
  stageScale: number;
  stagePos: { x: number; y: number };
  toolMode: ToolMode;
  selectedId: number | null;
  selectedVertexIdx: number | null;
  maskOpacity: number;
  segments: EditableSegment[];
  samPoints: PromptPoint[];
  samPendingPolygon: [number, number][] | null;
  drawPoints: [number, number][];
  mouseCanvasPos: [number, number] | null;
  onStagePosChange: (pos: { x: number; y: number }) => void;
  onWheel: (e: Konva.KonvaEventObject<WheelEvent>) => void;
  onStageClick: (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => void;
  onStageMouseMove: () => void;
  onSelectCoral: (id: number) => void;
  onAddVertexOnEdge: (segId: number, edgeIdx: number, pt: [number, number]) => void;
  onSelectVertex: (vIdx: number) => void;
  onVertexDrag: (segId: number, vIdx: number, pos: { x: number; y: number }) => void;
  onVertexDragEnd: () => void;
  onDeleteVertex: (segId: number, vIdx: number) => void;
}

export const EditorCanvas: React.FC<EditorCanvasProps> = ({
  containerRef,
  stageRef,
  htmlImage,
  imgDimensions,
  stageScale,
  stagePos,
  toolMode,
  selectedId,
  selectedVertexIdx,
  maskOpacity,
  segments,
  samPoints,
  samPendingPolygon,
  drawPoints,
  mouseCanvasPos,
  onStagePosChange,
  onWheel,
  onStageClick,
  onStageMouseMove,
  onSelectCoral,
  onAddVertexOnEdge,
  onSelectVertex,
  onVertexDrag,
  onVertexDragEnd,
  onDeleteVertex,
}) => {
  const [isSpaceDown, setIsSpaceDown] = useState<boolean>(false);
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const isPanningRef = useRef<boolean>(false);
  const didDragRef = useRef<boolean>(false);
  const panStartRef = useRef<{ clientX: number; clientY: number; stageX: number; stageY: number }>({
    clientX: 0,
    clientY: 0,
    stageX: 0,
    stageY: 0,
  });
  const panRafRef = useRef<number | null>(null);

  // Track Space key for Hand Tool pan mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement as HTMLElement;
      const isInput = activeEl && ["INPUT", "TEXTAREA", "SELECT"].includes(activeEl.tagName);
      if (isInput) return;
      if (e.code === "Space" && !e.repeat) {
        setIsSpaceDown(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        setIsSpaceDown(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // Window mousemove & mouseup listeners for buttery smooth, zero-latency canvas panning
  useEffect(() => {
    const handleWindowMouseMove = (e: MouseEvent) => {
      if (!isPanningRef.current || !stageRef.current) return;

      const dx = e.clientX - panStartRef.current.clientX;
      const dy = e.clientY - panStartRef.current.clientY;

      if (Math.hypot(dx, dy) > 3) {
        didDragRef.current = true;
      }

      const nextX = panStartRef.current.stageX + dx;
      const nextY = panStartRef.current.stageY + dy;

      if (panRafRef.current === null) {
        panRafRef.current = requestAnimationFrame(() => {
          if (stageRef.current) {
            stageRef.current.position({ x: nextX, y: nextY });
            stageRef.current.batchDraw();
          }
          panRafRef.current = null;
        });
      }
    };

    const handleWindowMouseUp = () => {
      if (isPanningRef.current) {
        isPanningRef.current = false;
        setIsPanning(false);

        if (panRafRef.current !== null) {
          cancelAnimationFrame(panRafRef.current);
          panRafRef.current = null;
        }

        if (stageRef.current) {
          const finalPos = stageRef.current.position();
          onStagePosChange({
            x: Math.round(finalPos.x),
            y: Math.round(finalPos.y),
          });
        }

        // Brief delay before resetting drag flag to prevent trailing click events
        setTimeout(() => {
          didDragRef.current = false;
        }, 60);
      }
    };

    window.addEventListener("mousemove", handleWindowMouseMove, { passive: true });
    window.addEventListener("mouseup", handleWindowMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", handleWindowMouseUp);
      if (panRafRef.current !== null) {
        cancelAnimationFrame(panRafRef.current);
        panRafRef.current = null;
      }
    };
  }, [onStagePosChange, stageRef]);

  if (!htmlImage) return null;

  // Determine when mouse down should initiate canvas panning
  const handleStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const isMiddleClick = e.evt.button === 1;
    const isLeftClick = e.evt.button === 0;
    const isSpacePan = isSpaceDown && isLeftClick;
    const isBgOrStage = e.target === stageRef.current || e.target.name() === "bg-image";
    const isCanvasPan = toolMode === "select" && isLeftClick && isBgOrStage;

    if (isMiddleClick || isSpacePan || isCanvasPan) {
      e.evt.preventDefault();
      isPanningRef.current = true;
      didDragRef.current = false;
      setIsPanning(true);

      const stage = stageRef.current;
      const currentX = stage ? stage.x() : stagePos.x;
      const currentY = stage ? stage.y() : stagePos.y;

      panStartRef.current = {
        clientX: e.evt.clientX,
        clientY: e.evt.clientY,
        stageX: currentX,
        stageY: currentY,
      };
    }
  };

  const handleSafeStageClick = (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (didDragRef.current || isPanningRef.current) {
      return;
    }
    onStageClick(e);
  };

  const getCursor = () => {
    if (isPanning) return "grabbing";
    if (isSpaceDown) return "grab";
    if (toolMode === "select") return "default";
    return "crosshair";
  };

  return (
    <Stage
      ref={stageRef}
      width={containerRef.current?.clientWidth || 800}
      height={containerRef.current?.clientHeight || 600}
      scaleX={stageScale}
      scaleY={stageScale}
      x={stagePos.x}
      y={stagePos.y}
      onMouseDown={handleStageMouseDown}
      onWheel={onWheel}
      onClick={handleSafeStageClick}
      onTap={handleSafeStageClick}
      onMouseMove={onStageMouseMove}
      style={{ cursor: getCursor() }}
    >
      {/* Background Reef Image - hit testing disabled for maximum render performance */}
      <Layer listening={false}>
        <KonvaImage
          name="bg-image"
          image={htmlImage}
          width={imgDimensions.w}
          height={imgDimensions.h}
        />
      </Layer>

      {/* Polygons & Annotations Layer - disabled during pan/spacebar for smooth 120fps motion */}
      <Layer
        listening={!isSpaceDown && !isPanning && toolMode === "select"}
        perfectDrawEnabled={false}
      >
        {segments.map((seg) => {
          if (!seg.visible || seg.polygon.length < 3) return null;
          const isSelected = seg.id === selectedId;
          const flatPoints = seg.polygon.flatMap((pt) => [pt[0], pt[1]]);
          const [cx, cy] = seg.polygon[0];

          return (
            <Group key={seg.id}>
              {/* Polygon fill & stroke */}
              <Line
                points={flatPoints}
                closed={true}
                fill={seg.color_hex}
                opacity={isSelected ? Math.min(maskOpacity + 0.2, 0.85) : maskOpacity}
                stroke={isSelected ? "#ffffff" : seg.color_hex}
                strokeWidth={isSelected ? 3 / stageScale : 2 / stageScale}
                shadowColor={isSelected ? "#000000" : undefined}
                shadowBlur={isSelected ? 6 : 0}
                perfectDrawEnabled={false}
                onClick={(e) => {
                  if (didDragRef.current || isPanningRef.current || isSpaceDown) {
                    e.cancelBubble = true;
                    return;
                  }
                  e.cancelBubble = true;
                  if (seg.id !== selectedId) {
                    onSelectCoral(seg.id);
                    return;
                  }
                  // If already selected, check if user clicked on/near edge to insert a vertex
                  const stage = stageRef.current;
                  if (stage && toolMode === "select") {
                    const transform = stage.getAbsoluteTransform().copy().invert();
                    const pointer = stage.getPointerPosition();
                    if (pointer) {
                      const pt = transform.point(pointer);
                      const uniquePts = getUniquePolygonPoints(seg.polygon);
                      const closest = findClosestEdge([pt.x, pt.y], uniquePts);
                      if (closest && closest.distance <= 12 / stageScale) {
                        onAddVertexOnEdge(seg.id, closest.edgeIdx, closest.projected);
                      }
                    }
                  }
                }}
                onDblClick={(e) => {
                  if (isSelected && toolMode === "select") {
                    e.cancelBubble = true;
                    const stage = stageRef.current;
                    if (stage) {
                      const transform = stage.getAbsoluteTransform().copy().invert();
                      const pointer = stage.getPointerPosition();
                      if (pointer) {
                        const pt = transform.point(pointer);
                        const uniquePts = getUniquePolygonPoints(seg.polygon);
                        const closest = findClosestEdge([pt.x, pt.y], uniquePts);
                        if (closest) {
                          onAddVertexOnEdge(seg.id, closest.edgeIdx, closest.projected);
                        }
                      }
                    }
                  }
                }}
              />

              {/* Floating ID badge */}
              <Group x={cx} y={cy}>
                <Rect
                  x={-24 / stageScale}
                  y={-14 / stageScale}
                  width={48 / stageScale}
                  height={16 / stageScale}
                  fill={isSelected ? "#0d7c85" : "#0f172a"}
                  stroke={isSelected ? "#ffffff" : seg.color_hex}
                  strokeWidth={1 / stageScale}
                  cornerRadius={3 / stageScale}
                />
                <KonvaText
                  x={-24 / stageScale}
                  y={-12 / stageScale}
                  width={48 / stageScale}
                  text={seg.id_str}
                  fontSize={10 / stageScale}
                  fontStyle="bold"
                  fontFamily="monospace"
                  fill="#ffffff"
                  align="center"
                />
              </Group>

              {/* Interactive Vertex Handles & Midpoint (+) Handles when Selected */}
              {isSelected &&
                toolMode === "select" &&
                (() => {
                  const uniquePts = getUniquePolygonPoints(seg.polygon);
                  return (
                    <>
                      {/* Midpoint (+) handles on each edge between adjacent vertices */}
                      {uniquePts.map((pt, vIdx) => {
                        const nextPt = uniquePts[(vIdx + 1) % uniquePts.length];
                        const midX = (pt[0] + nextPt[0]) / 2;
                        const midY = (pt[1] + nextPt[1]) / 2;
                        const edgeDist = Math.hypot(nextPt[0] - pt[0], nextPt[1] - pt[1]);

                        // Only show if the segment is reasonably visible on screen
                        if (edgeDist * stageScale < 14) return null;

                        return (
                          <Group key={`mid-${vIdx}`} x={midX} y={midY}>
                            <Circle
                              radius={4.5 / stageScale}
                              fill="#ffffff"
                              stroke="#0d7c85"
                              strokeWidth={1.5 / stageScale}
                              opacity={0.85}
                              onClick={(e) => {
                                e.cancelBubble = true;
                                onAddVertexOnEdge(seg.id, vIdx, [
                                  Math.round(midX),
                                  Math.round(midY),
                                ]);
                              }}
                              onMouseEnter={(e) => {
                                e.target.to({
                                  scaleX: 1.35,
                                  scaleY: 1.35,
                                  opacity: 1,
                                  duration: 0.1,
                                });
                                const container = stageRef.current?.container();
                                if (container) container.style.cursor = "copy";
                              }}
                              onMouseLeave={(e) => {
                                e.target.to({
                                  scaleX: 1,
                                  scaleY: 1,
                                  opacity: 0.85,
                                  duration: 0.1,
                                });
                                const container = stageRef.current?.container();
                                if (container) container.style.cursor = "default";
                              }}
                            />
                          </Group>
                        );
                      })}

                      {/* Main Draggable Vertex Anchors */}
                      {uniquePts.map((pt, vIdx) => {
                        const isVertexActive = selectedVertexIdx === vIdx;
                        return (
                          <Circle
                            key={`vert-${vIdx}`}
                            x={pt[0]}
                            y={pt[1]}
                            radius={(isVertexActive ? 7.5 : 5.5) / stageScale}
                            fill={isVertexActive ? "#00e5ff" : "#ffffff"}
                            stroke={isVertexActive ? "#000000" : "#0f172a"}
                            strokeWidth={(isVertexActive ? 2.5 : 1.5) / stageScale}
                            shadowColor="#000000"
                            shadowBlur={isVertexActive ? 6 : 2}
                            draggable
                            onDragStart={() => {
                              onSelectVertex(vIdx);
                            }}
                            onDragMove={(e) => {
                              onVertexDrag(seg.id, vIdx, {
                                x: e.target.x(),
                                y: e.target.y(),
                              });
                            }}
                            onDragEnd={() => {
                              onVertexDragEnd();
                            }}
                            onClick={(e) => {
                              e.cancelBubble = true;
                              if (e.evt.altKey) {
                                onDeleteVertex(seg.id, vIdx);
                              } else {
                                onSelectVertex(vIdx);
                              }
                            }}
                            onContextMenu={(e) => {
                              e.evt.preventDefault();
                              e.cancelBubble = true;
                              onDeleteVertex(seg.id, vIdx);
                            }}
                            onDblClick={(e) => {
                              e.cancelBubble = true;
                              onDeleteVertex(seg.id, vIdx);
                            }}
                            onMouseEnter={(e) => {
                              const container = stageRef.current?.container();
                              if (container) {
                                container.style.cursor = e.evt.altKey ? "not-allowed" : "grab";
                              }
                            }}
                            onMouseLeave={() => {
                              const container = stageRef.current?.container();
                              if (container) {
                                container.style.cursor = "default";
                              }
                            }}
                          />
                        );
                      })}
                    </>
                  );
                })()}
            </Group>
          );
        })}

        {/* SAM Prompt Points Pins (Green + for Coral, Red - for Background) */}
        {samPoints.map((pt, idx) => {
          const isPositive = pt.label === 1;
          const pinColor = isPositive ? "#10b981" : "#ef4444";
          return (
            <Group key={idx} x={pt.x} y={pt.y}>
              <Circle
                radius={10 / stageScale}
                fill={pinColor}
                opacity={0.25}
              />
              <Circle
                radius={7 / stageScale}
                fill={pinColor}
                stroke="#ffffff"
                strokeWidth={1.5 / stageScale}
                shadowColor="#000000"
                shadowBlur={4}
              />
              <KonvaText
                x={-6 / stageScale}
                y={-5 / stageScale}
                width={12 / stageScale}
                text={isPositive ? "+" : "−"}
                fontSize={10 / stageScale}
                fontStyle="bold"
                fontFamily="sans-serif"
                fill="#ffffff"
                align="center"
              />
            </Group>
          );
        })}

        {/* SAM Predicted Polygon Preview */}
        {samPendingPolygon && samPendingPolygon.length >= 3 && (
          <Line
            points={samPendingPolygon.flatMap((p) => [p[0], p[1]])}
            closed={true}
            fill="#10b981"
            opacity={0.45}
            stroke="#ffffff"
            strokeWidth={2.5 / stageScale}
            dash={[4 / stageScale, 4 / stageScale]}
            perfectDrawEnabled={false}
          />
        )}

        {/* Manual Polygon Drawing in Progress */}
        {drawPoints.length > 0 && (
          <Group>
            <Line
              points={[
                ...drawPoints.flatMap((p) => [p[0], p[1]]),
                ...(mouseCanvasPos ? [mouseCanvasPos[0], mouseCanvasPos[1]] : []),
              ]}
              closed={false}
              stroke="#00e5ff"
              strokeWidth={2 / stageScale}
              dash={[3 / stageScale, 3 / stageScale]}
              perfectDrawEnabled={false}
            />
            {drawPoints.map((p, idx) => (
              <Circle
                key={idx}
                x={p[0]}
                y={p[1]}
                radius={idx === 0 ? 6 / stageScale : 4 / stageScale}
                fill={idx === 0 ? "#ffd600" : "#00e5ff"}
                stroke="#ffffff"
                strokeWidth={1.5 / stageScale}
              />
            ))}
          </Group>
        )}
      </Layer>
    </Stage>
  );
};
