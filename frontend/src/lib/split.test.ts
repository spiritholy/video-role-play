import { describe, expect, it } from "vitest";
import type { Playing, SplitInterval } from "../types";
import {
  MIN_SPLIT_SECONDS,
  NEW_SLOT_PREFIX,
  NEW_SLOT_TOKEN,
  SPLIT_EPS,
  isNewSlotTarget,
  splitIntervals,
  splitPlayBounds,
  toSplitResults,
  validateSplitDraft,
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

const iv = (start: number, end: number, i: number): SplitInterval => ({
  key: `iv${i}`,
  start,
  end,
});

describe("validateSplitDraft", () => {
  const two = [iv(10, 15, 0), iv(15, 20, 1)];

  it("整段没有切分时不能保存", () => {
    expect(validateSplitDraft([iv(10, 20, 0)], ["SPEAKER_01"], "SPEAKER_00")).toBe(
      "至少需要两个结果区间才能保存",
    );
  });

  it("结果区间窄于 0.1 秒时拒绝", () => {
    expect(
      validateSplitDraft([iv(10, 10.05, 0), iv(10.05, 20, 1)], ["SPEAKER_01", "SPEAKER_00"], "SPEAKER_00"),
    ).toBe("每个结果区间至少 0.1 秒");
  });

  it("有区间未指定说话人时拒绝", () => {
    expect(validateSplitDraft(two, ["SPEAKER_01", ""], "SPEAKER_00")).toBe(
      "所有结果区间都必须指定说话人",
    );
    expect(validateSplitDraft(two, ["SPEAKER_01"], "SPEAKER_00")).toBe(
      "所有结果区间都必须指定说话人",
    );
  });

  it("全部区间仍指向原说话人时拒绝", () => {
    expect(validateSplitDraft(two, ["SPEAKER_00", "SPEAKER_00"], "SPEAKER_00")).toBe(
      "至少一个结果区间需要改派为非原说话人才能保存",
    );
  });

  it("既有切分又有改派时通过", () => {
    expect(validateSplitDraft(two, ["SPEAKER_00", "SPEAKER_01"], "SPEAKER_00")).toBeNull();
    expect(validateSplitDraft(two, ["new-slot-1", "SPEAKER_00"], "SPEAKER_00")).toBeNull();
  });
});

describe("toSplitResults", () => {
  it("草稿槽位映射为 __new__ + slot，现有说话人直接透传", () => {
    const results = toSplitResults([iv(10, 15, 0), iv(15, 20, 1)], [
      "new-slot-2",
      "SPEAKER_01",
    ]);
    expect(results).toEqual([
      { start: 10, end: 15, target: NEW_SLOT_TOKEN, slot: "new-slot-2" },
      { start: 15, end: 20, target: "SPEAKER_01" },
    ]);
  });

  it("同一槽位的多个区间保留同一个 slot 标识", () => {
    const results = toSplitResults(
      [iv(10, 13, 0), iv(13, 16, 1), iv(16, 20, 2)],
      ["new-slot-1", "SPEAKER_00", "new-slot-1"],
    );
    expect(results[0].slot).toBe(results[2].slot);
    expect(results[1].slot).toBeUndefined();
  });
});

describe("splitPlayBounds", () => {
  const playing = (
    start: number,
    end: number,
    speaker = "SPEAKER_00",
    taskId = "t1",
  ): Playing => ({ taskId, speaker, start, end });
  const intervals = [iv(10, 15, 0), iv(15, 20, 1)];

  it("整段试听覆盖原时段起止", () => {
    expect(splitPlayBounds(playing(10, 20), "t1", "SPEAKER_00", 10, 20, intervals)).toEqual({
      from: 10,
      to: 20,
    });
  });

  it("区间试听只在对应区间内移动", () => {
    expect(splitPlayBounds(playing(15, 20), "t1", "SPEAKER_00", 10, 20, intervals)).toEqual({
      from: 15,
      to: 20,
    });
  });

  it("任务、说话人不匹配或区间不属于当前草稿时没有播放条", () => {
    expect(splitPlayBounds(null, "t1", "SPEAKER_00", 10, 20, intervals)).toBeNull();
    expect(splitPlayBounds(playing(10, 20, "SPEAKER_01"), "t1", "SPEAKER_00", 10, 20, intervals)).toBeNull();
    expect(splitPlayBounds(playing(10, 20, "SPEAKER_00", "t2"), "t1", "SPEAKER_00", 10, 20, intervals)).toBeNull();
    expect(splitPlayBounds(playing(12, 18), "t1", "SPEAKER_00", 10, 20, intervals)).toBeNull();
  });
});
