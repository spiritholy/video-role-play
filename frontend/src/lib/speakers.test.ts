import { describe, expect, it } from "vitest";
import { SPEAKER_COLORS, shownName, speakerColor } from "./speakers";
import type { TimelineSpeaker } from "../types";

const sp = (label: string, name: string | null = null): TimelineSpeaker => ({
  label,
  name,
  speech_time: 0,
  segments: [],
});

describe("speakerColor", () => {
  it("按说话人在列表中的位置取色，超出配色数取模循环", () => {
    const labels = Array.from({ length: SPEAKER_COLORS.length + 1 }, (_, i) =>
      `SPEAKER_${String(i).padStart(2, "0")}`,
    );
    const speakers = labels.map((l) => sp(l));
    expect(speakerColor(speakers, speakers[0].label)).toBe(SPEAKER_COLORS[0]);
    expect(speakerColor(speakers, speakers[3].label)).toBe(SPEAKER_COLORS[3]);
    expect(speakerColor(speakers, speakers[2].label)).toBe(SPEAKER_COLORS[2]);
    expect(speakerColor(speakers, speakers[8].label)).toBe(SPEAKER_COLORS[0]);
  });
});

describe("shownName", () => {
  const speakers = [sp("SPEAKER_00", "旁白"), sp("SPEAKER_01")];

  it("手工命名优先于时间轴带来的名字", () => {
    expect(shownName({ SPEAKER_00: "男主" }, speakers[0])).toBe("男主");
  });

  it("无手工命名时用说话人命名 (Speaker Name)", () => {
    expect(shownName({}, speakers[0])).toBe("旁白");
  });

  it("两者都没有时回落到编号标签", () => {
    expect(shownName({}, speakers[1])).toBe("SPEAKER_01");
  });
});
