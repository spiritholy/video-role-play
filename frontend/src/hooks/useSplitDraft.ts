// 时段切分草稿状态机：间隔线、结果区间分配、草稿内新建槽位与指针拖拽。
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  MIN_SPLIT_SECONDS,
  SPLIT_EPS,
  isNewSlotTarget,
  splitIntervals,
} from "../lib/split";
import type { TimelineSegment } from "../types";

export function useSplitDraft({
  segment,
  originalSpeaker,
  editable,
  onDirtyChange,
}: {
  segment: TimelineSegment;
  originalSpeaker: string;
  editable: boolean;
  onDirtyChange: (dirty: boolean) => void;
}) {
  const start = segment.start;
  const end = segment.end;
  const [markers, setMarkers] = useState<number[]>([]);
  // 与结果区间一一对应的目标数组：现有说话人标签，或草稿新建槽位 id。
  const [targets, setTargets] = useState<string[]>([originalSpeaker]);
  const [slots, setSlots] = useState<string[]>([]);
  const [nextSlot, setNextSlot] = useState(1);
  const [hovered, setHovered] = useState<number | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trackWidth, setTrackWidth] = useState(0);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const dragGuard = useRef(false);

  const intervals = splitIntervals(start, end, markers);
  const dirty =
    markers.length > 0 ||
    slots.length > 0 ||
    targets.some((t) => t !== originalSpeaker);
  useEffect(() => {
    onDirtyChange(dirty);
  }, [dirty, onDirtyChange]);

  const pct = (t: number) =>
    `${(((t - start) / (end - start)) * 100).toFixed(3)}%`;
  // 切分线定位：按 track 像素取整，避免亚像素渲染导致线条粗细/明暗不一致。
  const markerLeft = (t: number): string =>
    trackWidth > 0
      ? `${Math.round(((t - start) / (end - start)) * trackWidth)}px`
      : pct(t);
  useLayoutEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const update = () => setTrackWidth(el.clientWidth);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // 结果区间宽度按「时长」占选中时段的比例换算，不能复用 pct（pct 的入参是绝对时间）。
  const durPct = (d: number) => `${((d / (end - start)) * 100).toFixed(3)}%`;

  function timeFromClientX(clientX: number): number | null {
    const el = trackRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    return start + frac * (end - start);
  }

  function addMarker(time: number) {
    if (!editable || time <= start || time >= end) return;
    const next = [...markers, time].sort((a, b) => a - b);
    const points = [start, ...next, end];
    for (let i = 1; i < points.length - 1; i++) {
      if (points[i] - points[i - 1] < MIN_SPLIT_SECONDS - SPLIT_EPS) {
        setError("间隔线间距不足 0.1 秒，无法在该位置添加");
        return;
      }
    }
    const insertAt = intervals.findIndex((iv) => time <= iv.end + SPLIT_EPS);
    const idx = insertAt < 0 ? intervals.length - 1 : insertAt;
    const newTargets = [...targets];
    // 新区间继承被切开区间的目标
    newTargets.splice(idx + 1, 0, targets[idx] ?? originalSpeaker);
    setMarkers(next);
    setTargets(newTargets);
    setError(null);
  }

  function deleteMarker(markerIndex: number) {
    if (!editable) return;
    setMarkers((prev) => prev.filter((_, i) => i !== markerIndex));
    setTargets((prev) => {
      const next = [...prev];
      // 合并相邻两个区间，保留左侧目标
      next.splice(markerIndex + 1, 1);
      return next;
    });
    setHovered(null);
    setError(null);
  }

  function discardDraft() {
    setMarkers([]);
    setTargets([originalSpeaker]);
    setSlots([]);
    setNextSlot(1);
    setError(null);
  }

  function addSlot() {
    if (!editable) return;
    setSlots((prev) => [...prev, `new-slot-${nextSlot}`]);
    setNextSlot((n) => n + 1);
    setError(null);
  }

  function markerBounds(index: number): { lower: number; upper: number } {
    const prev = index > 0 ? markers[index - 1] : start;
    const nextBound = index < markers.length - 1 ? markers[index + 1] : end;
    return { lower: prev + MIN_SPLIT_SECONDS, upper: nextBound - MIN_SPLIT_SECONDS };
  }

  function onMarkerPointerDown(
    event: ReactPointerEvent<HTMLDivElement>,
    index: number,
  ) {
    if (!editable) return;
    event.preventDefault();
    event.stopPropagation();
    dragGuard.current = true;
    setDragging(index);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onMarkerPointerMove(
    event: ReactPointerEvent<HTMLDivElement>,
    index: number,
  ) {
    if (dragging !== index) return;
    const time = timeFromClientX(event.clientX);
    if (time === null) return;
    const { lower, upper } = markerBounds(index);
    const clamped = Math.min(Math.max(time, lower), upper);
    if (Math.abs(clamped - markers[index]) > SPLIT_EPS) {
      dragGuard.current = true;
    }
    setMarkers((prev) => {
      const next = [...prev];
      next[index] = clamped;
      return next;
    });
    setError(null);
  }

  function onMarkerPointerUp(
    event: ReactPointerEvent<HTMLDivElement>,
    index: number,
  ) {
    if (dragging !== index) return;
    setDragging(null);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // 指针捕获已释放时忽略
    }
  }

  function onTrackClick(event: ReactMouseEvent<HTMLDivElement>) {
    if (!editable) return;
    if (dragGuard.current) {
      dragGuard.current = false;
      return;
    }
    const time = timeFromClientX(event.clientX);
    if (time !== null) addMarker(time);
  }

  function targetLabel(target: string): string {
    if (isNewSlotTarget(target)) {
      const n = slots.indexOf(target);
      return n >= 0 ? `新建说话人 ${n + 1}` : target;
    }
    return target;
  }

  return {
    start,
    end,
    markers,
    targets,
    slots,
    hovered,
    dragging,
    error,
    trackWidth,
    intervals,
    dirty,
    trackRef,
    dragGuard,
    setTargets,
    setHovered,
    setError,
    pct,
    durPct,
    markerLeft,
    addMarker,
    deleteMarker,
    discardDraft,
    addSlot,
    markerBounds,
    onMarkerPointerDown,
    onMarkerPointerMove,
    onMarkerPointerUp,
    onTrackClick,
    targetLabel,
  };
}

// 切分草稿的全部状态与操作，供切分轨道与结果区间列表两个子组件共享。
export type SplitDraft = ReturnType<typeof useSplitDraft>;
