import { describe, expect, it } from "vitest";
import { deriveSamples } from "./samples";
import type { Stage, Transcripts } from "../types";

const stage = (
  status: Stage["status"],
): Stage => ({
  id: "transcribe",
  name: "台词转写",
  status,
  error: null,
  artifacts: [],
  progress: null,
  outputs: {},
});

const transcripts = (samples: Transcripts["samples"]): Transcripts => ({
  model: "sensevoice",
  samples,
});

describe("deriveSamples", () => {
  it("转写数据未到达时按阶段状态给提示", () => {
    expect(deriveSamples(undefined, stage("pending"))).toEqual({
      samplesHint: "样本转写中…",
      samplesBySpeaker: {},
    });
    expect(deriveSamples(undefined, stage("running"))).toEqual({
      samplesHint: "样本转写中…",
      samplesBySpeaker: {},
    });
    expect(deriveSamples(undefined, stage("done"))).toEqual({
      samplesHint: "正在加载样本台词…",
      samplesBySpeaker: {},
    });
  });

  it("转写阶段缺失或尚未开始时无提示", () => {
    expect(deriveSamples(undefined, undefined).samplesHint).toBeNull();
    expect(deriveSamples(undefined, stage("failed")).samplesHint).toBeNull();
  });

  it("转写加载失败时提示失败", () => {
    expect(deriveSamples("error", stage("done"))).toEqual({
      samplesHint: "样本台词加载失败",
      samplesBySpeaker: {},
    });
  });

  it("有样本时按说话人分组，保留组内顺序", () => {
    const { samplesHint, samplesBySpeaker } = deriveSamples(
      transcripts([
        { speaker: "SPEAKER_01", start: 5, end: 8, text: "b" },
        { speaker: "SPEAKER_00", start: 1, end: 3, text: "a" },
        { speaker: "SPEAKER_01", start: 9, end: 11, text: "c" },
      ]),
      stage("done"),
    );
    expect(samplesHint).toBeNull();
    expect(Object.keys(samplesBySpeaker)).toEqual(["SPEAKER_01", "SPEAKER_00"]);
    expect(samplesBySpeaker.SPEAKER_01.map((s) => s.text)).toEqual(["b", "c"]);
  });
});
