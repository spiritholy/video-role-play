// 时段切分 (Segment Split) 的区间推导与新建说话人槽位协议。
import type { Playing, SplitInterval, SplitResult } from "../types";

// 切分结果区间最小宽度（秒），与后端校验保持一致。
export const MIN_SPLIT_SECONDS = 0.1;
export const SPLIT_EPS = 1e-9;

// 由间隔线推导相邻结果区间：起点/间隔线/终点顺序切分。
export function splitIntervals(start: number, end: number, markers: number[]): SplitInterval[] {
  const points = [start, ...markers, end];
  return points.slice(0, -1).map((a, i) => ({
    key: `iv${i}`,
    start: a,
    end: points[i + 1],
  }));
}

// 新建说话人槽位的 API 协议标记（与后端保持一致）。
export const NEW_SLOT_TOKEN = "__new__";
export const NEW_SLOT_PREFIX = "new-slot-";

export function isNewSlotTarget(target: string | undefined): boolean {
  return Boolean(target && target.startsWith(NEW_SLOT_PREFIX));
}

// 保存前的完整校验：切分必须至少两个结果区间；整段改说话人只走改派对话框。
export function validateSplitDraft(
  intervals: SplitInterval[],
  targets: string[],
  originalSpeaker: string,
): string | null {
  if (intervals.length < 2) {
    return "至少需要两个结果区间才能保存";
  }
  for (const iv of intervals) {
    if (iv.end - iv.start < MIN_SPLIT_SECONDS - SPLIT_EPS) {
      return "每个结果区间至少 0.1 秒";
    }
  }
  for (let i = 0; i < intervals.length; i++) {
    if (!targets[i]) return "所有结果区间都必须指定说话人";
  }
  if (intervals.every((_, i) => targets[i] === originalSpeaker)) {
    return "至少一个结果区间需要改派为非原说话人才能保存";
  }
  return null;
}

// 组装保存协议：草稿内的新建槽位映射为 __new__ + slot，同一槽位的多个区间
// 后端归并为同一个说话人标签。
export function toSplitResults(
  intervals: SplitInterval[],
  targets: string[],
): SplitResult[] {
  return intervals.map((iv, i) => {
    const target = targets[i];
    if (isNewSlotTarget(target)) {
      return { start: iv.start, end: iv.end, target: NEW_SLOT_TOKEN, slot: target };
    }
    return { start: iv.start, end: iv.end, target };
  });
}

// 切分面板的移动播放条范围：整段试听覆盖原时段，区间试听只在对应区间内移动。
export function splitPlayBounds(
  playing: Playing | null,
  taskId: string,
  originalSpeaker: string,
  start: number,
  end: number,
  intervals: SplitInterval[],
): { from: number; to: number } | null {
  if (
    playing === null ||
    playing.taskId !== taskId ||
    playing.speaker !== originalSpeaker
  ) {
    return null;
  }
  if (playing.start === start && playing.end === end) {
    return { from: start, to: end };
  }
  const iv = intervals.find(
    (x) => x.start === playing.start && x.end === playing.end,
  );
  return iv ? { from: iv.start, to: iv.end } : null;
}
