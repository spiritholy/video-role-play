// 声纹校验 (Voiceprint Check) 的展示口径：任务上正在运行的状态优先于本地缓存，
// 建议文案只做呈现，不改动任何数据。
import type { Task, VoiceprintCheckState, VoiceprintSuggestion } from "../types";

export function effectiveCheck(
  task: Task,
  checkStates: Record<string, VoiceprintCheckState>,
): VoiceprintCheckState | null {
  const running =
    task.voiceprint_check &&
    ["queued", "running"].includes(task.voiceprint_check.state);
  return running ? task.voiceprint_check! : checkStates[task.id] ?? null;
}

export function suggestionText(s: VoiceprintSuggestion): string {
  const pct = (v?: number) => (v == null ? "" : ` · 相似度 ${Math.round(v * 100)}%`);
  if (s.type === "rename") return `建议把 ${s.label} 命名为「${s.name}」${pct(s.similarity)}`;
  if (s.type === "merge") return `建议合并同名簇 ${s.labels?.join("、")} 为 ${s.canonical}`;
  return `建议把 ${s.start?.toFixed(1)}s–${s.end?.toFixed(1)}s 从 ${s.from_label} 改派给「${s.name}」${pct(s.similarity)}`;
}
