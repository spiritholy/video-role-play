import { describe, expect, it } from "vitest";
import {
  MIN_SPLIT_SECONDS,
  NEW_SLOT_PREFIX,
  NEW_SLOT_TOKEN,
  SPLIT_EPS,
  isNewSlotTarget,
  splitIntervals,
} from "./split";

describe("splitIntervals", () => {
  it("无间隔线时整段作为一个结果区间", () => {
    expect(splitIntervals(10, 20, [])).toEqual([
      { key: "iv0", start: 10, end: 20 },
    ]);
  });

  it("一条间隔线把时段拆成相邻两段，切点同时是左段终点与右段起点", () => {
    expect(splitIntervals(10, 20, [15])).toEqual([
      { key: "iv0", start: 10, end: 15 },
      { key: "iv1", start: 15, end: 20 },
    ]);
  });

  it("多条间隔线按顺序拆分，首尾锚定原时段边界", () => {
    const intervals = splitIntervals(10, 20, [12.5, 17]);
    expect(intervals.map((i) => i.key)).toEqual(["iv0", "iv1", "iv2"]);
    expect(intervals[0].start).toBe(10);
    expect(intervals[intervals.length - 1].end).toBe(20);
    expect(intervals.map((i) => [i.start, i.end])).toEqual([
      [10, 12.5],
      [12.5, 17],
      [17, 20],
    ]);
  });
});

describe("新建说话人槽位协议", () => {
  it("常量与后端约定一致", () => {
    expect(NEW_SLOT_TOKEN).toBe("__new__");
    expect(NEW_SLOT_PREFIX).toBe("new-slot-");
    expect(MIN_SPLIT_SECONDS).toBeCloseTo(0.1, 10);
    expect(SPLIT_EPS).toBe(1e-9);
  });

  it("只认 new-slot- 前缀为草稿槽位", () => {
    expect(isNewSlotTarget("new-slot-1")).toBe(true);
    expect(isNewSlotTarget("SPEAKER_00")).toBe(false);
    expect(isNewSlotTarget(NEW_SLOT_TOKEN)).toBe(false);
    expect(isNewSlotTarget(undefined)).toBe(false);
  });
});
