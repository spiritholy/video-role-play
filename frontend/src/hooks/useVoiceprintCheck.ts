// 按需声纹校验 (Voiceprint Check)：显式触发，产出建议但不直接改数据；
// 建议逐条经接受/忽略后才并入人工校正。校验状态按任务键控。
import { useCallback, useRef, useState } from "react";
import { errorMessage } from "../api/client";
import {
  deleteVoiceprintCheck,
  getVoiceprintCheck,
  postSuggestionDecision,
  postVoiceprintCheck,
} from "../api/voiceprints";
import { workspaceKey } from "../lib/workspace";
import type { Task, TimelineUpdate, VoiceprintCheckState } from "../types";

export interface VoiceprintCheck {
  checkStates: Record<string, VoiceprintCheckState>;
  checkError: string | null;
  decisionBusy: Record<string, boolean>;
  maybeFetchCheck: (t: Task) => Promise<void>;
  startVoiceprintCheck: (t: Task) => Promise<void>;
  cancelVoiceprintCheck: (t: Task) => Promise<void>;
  decideSuggestion: (
    t: Task,
    index: number,
    decision: "accept" | "ignore",
  ) => Promise<void>;
}

export function useVoiceprintCheck(deps: {
  applyTimeline: (data: TimelineUpdate) => void;
}): VoiceprintCheck {
  const { applyTimeline } = deps;
  const [checkStates, setCheckStates] = useState<Record<string, VoiceprintCheckState>>({});
  const [checkError, setCheckError] = useState<string | null>(null);
  const [decisionBusy, setDecisionBusy] = useState<Record<string, boolean>>({});
  const requestedChecks = useRef<Set<string>>(new Set());

  const maybeFetchCheck = useCallback(async (t: Task) => {
    if (!t.stages.some((s) => s.id === "diarize" && s.status === "done")) return;
    const active = t.voiceprint_check && ["queued", "running"].includes(t.voiceprint_check.state);
    if (!active && requestedChecks.current.has(t.id)) return;
    requestedChecks.current.add(t.id);
    const result = await getVoiceprintCheck(t.id);
    // 失败时不动状态，下一轮轮询重试
    if (result.ok) {
      const data = result.data;
      setCheckStates((prev) => ({ ...prev, [t.id]: data }));
    }
  }, []);

  const startVoiceprintCheck = useCallback(async (t: Task) => {
    setCheckError(null);
    const result = await postVoiceprintCheck(t.id);
    if (!result.ok) {
      setCheckError(errorMessage(result, "声纹校验启动失败"));
      return;
    }
    const check = result.data.voiceprint_check;
    if (check) {
      setCheckStates((prev) => ({
        ...prev,
        [t.id]: { ...check, result: null },
      }));
    }
  }, []);

  const cancelVoiceprintCheck = useCallback(async (t: Task) => {
    setCheckError(null);
    const result = await deleteVoiceprintCheck(t.id);
    if (!result.ok) {
      setCheckError(errorMessage(result, "取消声纹校验失败"));
    }
  }, []);

  const decideSuggestion = useCallback(
    async (t: Task, index: number, decision: "accept" | "ignore") => {
      const key = `${t.id}:${index}`;
      if (decisionBusy[key]) return;
      setDecisionBusy((prev) => ({ ...prev, [key]: true }));
      setCheckError(null);
      const result = await postSuggestionDecision(t.id, index, decision);
      if (!result.ok) {
        setCheckError(errorMessage(result, "建议裁决失败"));
      } else {
        const data = result.data;
        const timeline = data.timeline;
        setCheckStates((prev) => ({
          ...prev,
          [t.id]: {
            ...(prev[t.id] ?? { state: "completed", progress: 1, error: null }),
            state: "completed",
            progress: 1,
            error: null,
            result: data.result,
          },
        }));
        applyTimeline({ taskId: t.id, workspaceKey: workspaceKey(t), timeline });
      }
      setDecisionBusy((prev) => ({ ...prev, [key]: false }));
    },
    [applyTimeline, decisionBusy],
  );

  return {
    checkStates,
    checkError,
    decisionBusy,
    maybeFetchCheck,
    startVoiceprintCheck,
    cancelVoiceprintCheck,
    decideSuggestion,
  };
}
