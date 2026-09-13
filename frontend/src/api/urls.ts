// 后端音频产物的取用地址（产物路径可能带子目录，逐段转义）。
import type { TimelineSegment } from "../types";

export function audioUrl(taskId: string, artifact: string): string {
  return `/api/tasks/${taskId}/audio/${
    artifact.split("/").map(encodeURIComponent).join("/")
  }`;
}

// 试听某个说话人时段：后端按 start/end 切片返回音频。
export function audioSegmentUrl(
  taskId: string,
  seg: Pick<TimelineSegment, "start" | "end">,
): string {
  return `/api/tasks/${taskId}/audio-segment?start=${seg.start}&end=${seg.end}`;
}
