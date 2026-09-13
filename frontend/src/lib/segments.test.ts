import { describe, expect, it } from "vitest";
import { firstSegment, isSegmentActive, orderedSegmentsOf } from "./segments";
import type { Playing, Timeline, TimelineSegment, TimelineSpeaker } from "../types";

const seg = (index: number, start: number, end: number): TimelineSegment => ({
  index,
  start,
  end,
});

const speaker = (label: string, segments: TimelineSegment[]): TimelineSpeaker => ({
  label,
  name: null,
  speech_time: 0,
  segments,
});

const timeline = (speakers: TimelineSpeaker[]): Timeline => ({
  model: "pyannote",
  num_speakers: speakers.length,
  duration: 100,
  speakers,
  correction: {
    state: "none",
    revision: 0,
    message: null,
    editable: true,
    locked_reason: null,
  },
});

const playing: Playing = {
  taskId: "t1",
  speaker: "SPEAKER_00",
  start: 10,
  end: 12,
};

describe("isSegmentActive", () => {
  it("任务、说话人、起止时间四项全中才算正在试听", () => {
    expect(isSegmentActive("t1", playing, "SPEAKER_00", seg(0, 10, 12))).toBe(true);
    expect(isSegmentActive("t2", playing, "SPEAKER_00", seg(0, 10, 12))).toBe(false);
    expect(isSegmentActive("t1", playing, "SPEAKER_01", seg(0, 10, 12))).toBe(false);
    expect(isSegmentActive("t1", playing, "SPEAKER_00", seg(0, 10, 13))).toBe(false);
    expect(isSegmentActive("t1", playing, "SPEAKER_00", seg(0, 9, 12))).toBe(false);
  });

  it("没有播放中的时段时一律为 false", () => {
    expect(isSegmentActive("t1", null, "SPEAKER_00", seg(0, 10, 12))).toBe(false);
  });
});

describe("orderedSegmentsOf", () => {
  it("跨说话人按起始时间排序，起始相同再按结束时间", () => {
    const t = timeline([
      speaker("SPEAKER_00", [seg(0, 30, 40), seg(1, 10, 12)]),
      speaker("SPEAKER_01", [seg(0, 10, 20), seg(1, 5, 6)]),
    ]);
    expect(orderedSegmentsOf(t).map((i) => [i.speaker, i.segment.start, i.segment.end])).toEqual([
      ["SPEAKER_01", 5, 6],
      ["SPEAKER_00", 10, 12],
      ["SPEAKER_01", 10, 20],
      ["SPEAKER_00", 30, 40],
    ]);
  });

  it("时段全部删除后为空数组", () => {
    expect(orderedSegmentsOf(timeline([speaker("SPEAKER_00", [])]))).toEqual([]);
  });
});

describe("firstSegment", () => {
  it("返回说话人列表中第一个有时段的说话人的首段（不按时间排序）", () => {
    const t = timeline([
      speaker("SPEAKER_00", []),
      speaker("SPEAKER_01", [seg(3, 50, 60), seg(4, 70, 80)]),
    ]);
    expect(firstSegment(t)).toEqual({
      speaker: "SPEAKER_01",
      segment: seg(3, 50, 60),
    });
  });

  it("没有任何时段时返回 null", () => {
    expect(firstSegment(timeline([speaker("SPEAKER_00", [])]))).toBeNull();
    expect(firstSegment(timeline([]))).toBeNull();
  });
});
