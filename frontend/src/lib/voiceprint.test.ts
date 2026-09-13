import { describe, expect, it } from "vitest";
import { effectiveCheck, suggestionText } from "./voiceprint";
import type { Task, VoiceprintCheckState, VoiceprintSuggestion } from "../types";

const check = (
  state: VoiceprintCheckState["state"],
  extra?: Partial<VoiceprintCheckState>,
): VoiceprintCheckState => ({
  state,
  progress: 0,
  error: null,
  ...extra,
});

const task = (
  id: string,
  voiceprintCheck?: VoiceprintCheckState | null,
): Task => ({
  id,
  video_path: "/v/a.mkv",
  audio_track: null,
  state: "completed",
  state_label: "已完成",
  error: null,
  stages: [],
  output_path: null,
  workspace_dir: null,
  erase: null,
  voiceprint_check: voiceprintCheck,
  created_at: "2026-01-01",
});

describe("suggestionText", () => {
  it("命名建议带相似度", () => {
    const s: VoiceprintSuggestion = {
      type: "rename",
      label: "SPEAKER_00",
      name: "小明",
      similarity: 0.87,
    };
    expect(suggestionText(s)).toBe("建议把 SPEAKER_00 命名为「小明」 · 相似度 87%");
  });

  it("命名建议无相似度时省略后缀", () => {
    expect(
      suggestionText({ type: "rename", label: "SPEAKER_01", name: "小红" }),
    ).toBe("建议把 SPEAKER_01 命名为「小红」");
  });

  it("合并建议列出同名簇与规范名", () => {
    expect(
      suggestionText({
        type: "merge",
        labels: ["SPEAKER_00", "SPEAKER_02"],
        canonical: "SPEAKER_00",
      }),
    ).toBe("建议合并同名簇 SPEAKER_00、SPEAKER_02 为 SPEAKER_00");
  });

  it("改派建议给出起止与目标名字", () => {
    expect(
      suggestionText({
        type: "reassign",
        start: 1.234,
        end: 5.6,
        from_label: "SPEAKER_00",
        name: "小刚",
        similarity: 0.5,
      }),
    ).toBe("建议把 1.2s–5.6s 从 SPEAKER_00 改派给「小刚」 · 相似度 50%");
  });
});

describe("effectiveCheck", () => {
  it("任务上排队/运行中的状态优先于本地缓存", () => {
    const running = check("running", { progress: 0.4 });
    const cached = check("completed");
    expect(effectiveCheck(task("t1", running), { t1: cached })).toBe(running);
    expect(effectiveCheck(task("t1", check("queued")), { t1: cached })).toEqual(
      check("queued"),
    );
  });

  it("任务上的终态不覆盖本地缓存", () => {
    const cached = check("completed");
    expect(effectiveCheck(task("t1", check("failed")), { t1: cached })).toBe(
      cached,
    );
  });

  it("任务上没有状态时回落到本地缓存，都没有则为 null", () => {
    const cached = check("cancelled");
    expect(effectiveCheck(task("t1"), { t1: cached })).toBe(cached);
    expect(effectiveCheck(task("t1"), {})).toBeNull();
  });
});
