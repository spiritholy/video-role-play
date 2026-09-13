// 说话人时段的排序、命中判定与最小可视宽度。
import type { Playing, Timeline, TimelineSegment } from "../types";

// 总览中时段的最小可视宽度：短时段至少 1px，便于发现；大量短时段超过
// 轨道容量时允许相邻时段视觉覆盖，不再靠横向滚动或整片放大来显示。
export const MIN_SEG_PX = 1;

export function isSegmentActive(
  taskId: string,
  playing: Playing | null,
  speaker: string,
  seg: Pick<TimelineSegment, "start" | "end">,
): boolean {
  return (
    playing !== null &&
    playing.taskId === taskId &&
    playing.speaker === speaker &&
    playing.start === seg.start &&
    playing.end === seg.end
  );
}

// 全片所有时段按时间排序（用于总览选中定位与保存后的停留定位）。
export function orderedSegmentsOf(
  timeline: Timeline,
): { speaker: string; segment: TimelineSegment }[] {
  const items: { speaker: string; segment: TimelineSegment }[] = [];
  for (const sp of timeline.speakers) {
    for (const seg of sp.segments) {
      items.push({ speaker: sp.label, segment: seg });
    }
  }
  items.sort(
    (a, b) => a.segment.start - b.segment.start || a.segment.end - b.segment.end,
  );
  return items;
}

export function firstSegment(
  timeline: Timeline,
): { speaker: string; segment: TimelineSegment } | null {
  for (const sp of timeline.speakers) {
    if (sp.segments.length > 0) {
      return { speaker: sp.label, segment: sp.segments[0] };
    }
  }
  return null;
}
