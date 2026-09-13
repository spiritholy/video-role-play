// 时段切分 (Segment Split) 的区间推导与新建说话人槽位协议。
import type { SplitInterval } from "../types";

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
