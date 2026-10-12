import { describe, expect, it } from "vitest";
import { dirName, fileName, fmtTime, stageSummary, trackLabel } from "./format";
import type { ProbeTrack, Stage, Task } from "../types";

const stage = (id: string, name: string, status: Stage["status"]): Stage => ({
  id,
  name,
  status,
  error: null,
  artifacts: [],
  progress: null,
  outputs: {},
});

const task = (state: Task["state"], stages: Stage[]): Task => ({
  id: "t1",
  video_path: "/videos/电影.mkv",
  audio_track: null,
  state,
  state_label: state,
  error: null,
  stages,
  output_path: null,
  workspace_dir: null,
  erase: null,
  created_at: "2026-09-13T10:00:00",
});

const track = (over: Partial<ProbeTrack> = {}): ProbeTrack => ({
  index: 0,
  stream_index: 1,
  codec: "aac",
  channels: null,
  channel_layout: null,
  language: null,
  ...over,
});

describe("fmtTime", () => {
  it("一分钟内保留一位小数的秒", () => {
    expect(fmtTime(0)).toBe("0.0s");
    expect(fmtTime(3.14)).toBe("3.1s");
    expect(fmtTime(59.9)).toBe("59.9s");
  });

  it("超过一分钟显示 分:秒", () => {
    expect(fmtTime(60)).toBe("1:00");
    expect(fmtTime(125.4)).toBe("2:05");
  });

  it("保留现状：秒部分四舍五入到 60 时不进位到下一分", () => {
    expect(fmtTime(119.6)).toBe("1:60");
  });
});

describe("fileName / dirName", () => {
  it("分别取最后一段与去掉最后一段", () => {
    expect(fileName("/a/b/c.mkv")).toBe("c.mkv");
    expect(fileName("C:\\videos\\y.mp4")).toBe("y.mp4");
    expect(fileName("plain.mp4")).toBe("plain.mp4");
    expect(dirName("/a/b/c.mkv")).toBe("/a/b");
    expect(dirName("C:\\videos\\y.mp4")).toBe("C:\\videos");
    expect(dirName("plain.mp4")).toBe("");
  });
});

describe("trackLabel", () => {
  it("只有编码时列出音轨序号与编码", () => {
    expect(trackLabel(track({ codec: null }))).toBe("音轨 1 · 未知编码");
  });

  it("有声道数但无声道布局时只列声道数", () => {
    expect(trackLabel(track({ channels: 2 }))).toBe("音轨 1 · aac · 2 声道");
  });

  it("声道布局与语言齐备时一并列出", () => {
    expect(
      trackLabel(
        track({ index: 1, channels: 6, channel_layout: "5.1", language: "zho" }),
      ),
    ).toBe("音轨 2 · aac · 5.1（6 声道） · 语言 zho");
  });
});

describe("stageSummary", () => {
  const three = [stage("separate", "人声分离", "done"), stage("diarize", "说话人分离", "done"), stage("encode_mux", "编码混流", "done")];

  it("已完成且擦除阶段完成时报擦除完成", () => {
    expect(stageSummary(task("completed", three))).toBe("擦除完成");
  });

  it("已完成但未擦除时报可选择要擦除的说话人", () => {
    const noErase = three.slice(0, 2);
    expect(stageSummary(task("completed", noErase))).toBe(
      "分析完成，可选择要擦除的说话人",
    );
  });

  it("运行中报当前 running 阶段", () => {
    const running = [
      stage("separate", "人声分离", "done"),
      stage("diarize", "说话人分离", "running"),
      stage("transcribe", "样本转写", "pending"),
    ];
    expect(stageSummary(task("running", running))).toBe("阶段 2/3 · 说话人分离");
  });

  it("没有 running 时报第一个未完成阶段，跳过 done 与 cancelled", () => {
    const mixed = [
      stage("separate", "人声分离", "done"),
      stage("diarize", "说话人分离", "cancelled"),
      stage("transcribe", "样本转写", "pending"),
    ];
    expect(stageSummary(task("cancelled", mixed))).toBe("阶段 3/3 · 样本转写");
  });

  it("非完成态但全部阶段已完成时报空串", () => {
    expect(stageSummary(task("queued", three))).toBe("");
  });
});
