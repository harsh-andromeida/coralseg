import { useState, useRef, useCallback } from "react";
import { EditableSegment } from "./types";
import { cloneSegments } from "./geometry";

export interface UseEditorHistoryProps {
  onStateRestored: (segments: EditableSegment[]) => void;
  markUnsaved: () => void;
}

export function useEditorHistory({ onStateRestored, markUnsaved }: UseEditorHistoryProps) {
  const historyRef = useRef<EditableSegment[][]>([]);
  const historyIdxRef = useRef<number>(-1);
  const [historyIdx, setHistoryIdx] = useState<number>(-1);
  const [historyLength, setHistoryLength] = useState<number>(0);

  const resetHistory = useCallback((initialList: EditableSegment[]) => {
    const cloned = cloneSegments(initialList);
    historyRef.current = [cloned];
    historyIdxRef.current = 0;
    setHistoryIdx(0);
    setHistoryLength(1);
  }, []);

  const pushHistory = useCallback(
    (newSegments: EditableSegment[]) => {
      const cloned = cloneSegments(newSegments);
      const curIdx = historyIdxRef.current;
      const nextHistory = historyRef.current.slice(0, curIdx + 1);
      nextHistory.push(cloned);

      if (nextHistory.length > 50) {
        nextHistory.shift();
      }

      historyRef.current = nextHistory;
      const nextIdx = nextHistory.length - 1;
      historyIdxRef.current = nextIdx;
      setHistoryIdx(nextIdx);
      setHistoryLength(nextHistory.length);
      markUnsaved();
    },
    [markUnsaved]
  );

  const undo = useCallback((): boolean => {
    if (historyIdxRef.current > 0) {
      const prevIdx = historyIdxRef.current - 1;
      historyIdxRef.current = prevIdx;
      setHistoryIdx(prevIdx);
      const restored = cloneSegments(historyRef.current[prevIdx]);
      onStateRestored(restored);
      markUnsaved();
      return true;
    }
    return false;
  }, [onStateRestored, markUnsaved]);

  const redo = useCallback((): boolean => {
    if (historyIdxRef.current < historyRef.current.length - 1) {
      const nextIdx = historyIdxRef.current + 1;
      historyIdxRef.current = nextIdx;
      setHistoryIdx(nextIdx);
      const restored = cloneSegments(historyRef.current[nextIdx]);
      onStateRestored(restored);
      markUnsaved();
      return true;
    }
    return false;
  }, [onStateRestored, markUnsaved]);

  return {
    historyIdx,
    historyLength,
    resetHistory,
    pushHistory,
    undo,
    redo,
    canUndo: historyIdx > 0,
    canRedo: historyIdx < historyLength - 1,
  };
}
