// 全片总览的固定 8 秒放大镜：窗口平移、时段几何与跟随鼠标的定位。
import type { TimelineSegment } from "../types";

// 放大镜窗口固定 8 秒；元素宽 360px（含左右各 1px 边框）。
export const MAGNIFIER_SECONDS = 8;
export const MAGNIFIER_WIDTH = 360;

export interface MagnifierWindow {
  windowStart: number;
  windowEnd: number;
  windowSeconds: number;
  centerPct: number;
}

// 固定 8 秒放大镜窗口：以悬停时间为中心，靠近片头/片尾时整体平移保持 8 秒。
export function magnifierWindow(
  duration: number,
  hoverTime: number | null,
): MagnifierWindow {
  const windowStart =
    duration > 0 && hoverTime !== null
      ? Math.min(
          Math.max(hoverTime - MAGNIFIER_SECONDS / 2, 0),
          Math.max(0, duration - MAGNIFIER_SECONDS),
        )
      : 0;
  const windowEnd =
    duration > 0 && hoverTime !== null
      ? Math.min(duration, windowStart + MAGNIFIER_SECONDS)
      : 0;
  const windowSeconds = Math.max(0, windowEnd - windowStart);
  const centerPct =
    windowSeconds > 0 && hoverTime !== null
      ? Math.min(
          100,
          Math.max(0, ((hoverTime - windowStart) / windowSeconds) * 100),
        )
      : 50;
  return { windowStart, windowEnd, windowSeconds, centerPct };
}

// 放大镜内时段的横向几何：只画落在窗口里的时段，起止按窗口换算成百分比。
export function magnifierSegmentGeometry(
  windowStart: number,
  windowEnd: number,
  windowSeconds: number,
): {
  inWindow: (seg: TimelineSegment) => boolean;
  leftPct: (seg: TimelineSegment) => number;
  widthPct: (seg: TimelineSegment) => number;
} {
  const inWindow = (seg: TimelineSegment) =>
    seg.end > windowStart && seg.start < windowEnd;
  const leftPct = (seg: TimelineSegment) =>
    Math.max(0, ((seg.start - windowStart) / windowSeconds) * 100);
  const rightPct = (seg: TimelineSegment) =>
    Math.min(100, ((seg.end - windowStart) / windowSeconds) * 100);
  return { inWindow, leftPct, widthPct: (seg) => Math.max(0, rightPct(seg) - leftPct(seg)) };
}

// 放大镜固定悬浮在鼠标上方：left = 鼠标 x - centerPct% × 放大镜 padding box 宽
// （360 - 左右边框 2px），使悬停时间的渲染位置（中线）始终对准鼠标。
export function magnifierPosition(
  clientX: number,
  clientY: number,
  centerPct: number,
  viewportWidth: number,
): { left: number; top: number } {
  const magnifierPaddingBox = MAGNIFIER_WIDTH - 2;
  return {
    left: Math.min(
      Math.max(clientX - (centerPct / 100) * magnifierPaddingBox, 8),
      viewportWidth - MAGNIFIER_WIDTH - 8,
    ),
    top: Math.max(clientY, 140),
  };
}

// 悬停位置 → 全片时间：以轨道列（时段实际绘制区域）宽度为基准，避免左侧名称
// 列造成偏差；越界夹取到轨道两端。
export function magnifierTimeFromClientX(
  clientX: number,
  containerRect: { left: number; width: number },
  trackRect: { left: number; width: number } | null,
  duration: number,
): number {
  let timePct = (clientX - containerRect.left) / containerRect.width;
  if (trackRect && trackRect.width > 0) {
    const trackX = Math.min(
      Math.max(clientX - trackRect.left, 0),
      trackRect.width,
    );
    timePct = trackX / trackRect.width;
  }
  return Math.min(duration, Math.max(0, timePct) * duration);
}
