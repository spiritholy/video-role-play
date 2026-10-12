// 1 秒轮询：任务列表、时间轴、样本转写、声纹校验状态、工作区与声纹库清单。
// refresh 的身份必须稳定，否则定时器会被每次渲染重建。
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchTranscripts } from "../api/analysis";
import { listTasks } from "../api/tasks";
import { listVoiceprints } from "../api/voiceprints";
import { listWorkspaces } from "../api/workspaces";
import type {
  Task,
  TranscriptsOrError,
  VoiceprintList,
  WorkspaceList,
} from "../types";

export interface ServerPolling {
  tasks: Task[];
  transcripts: Record<string, TranscriptsOrError>;
  workspaces: WorkspaceList | null;
  voiceprints: VoiceprintList | null;
  refresh: () => Promise<void>;
  refreshWorkspaces: () => Promise<void>;
  refreshVoiceprints: () => Promise<void>;
}

export function useServerPolling(deps: {
  loadTimeline: (t: Task) => Promise<void>;
  syncCheckState: (t: Task) => Promise<void>;
}): ServerPolling {
  const { loadTimeline, syncCheckState } = deps;
  const [tasks, setTasks] = useState<Task[]>([]);
  const [workspaces, setWorkspaces] = useState<WorkspaceList | null>(null);
  const [voiceprints, setVoiceprints] = useState<VoiceprintList | null>(null);
  const [transcripts, setTranscripts] = useState<Record<string, TranscriptsOrError>>({});
  const requestedTranscripts = useRef<Set<string>>(new Set());

  const maybeFetchTranscripts = useCallback(
    async (t: Task) => {
      if (!t.stages.some((s) => s.id === "transcribe" && s.status === "done")) return;
      if (requestedTranscripts.current.has(t.id)) return;
      requestedTranscripts.current.add(t.id);
      const result = await fetchTranscripts(t.id);
      setTranscripts((prev) => ({
        ...prev,
        [t.id]: result.ok ? result.data : "error",
      }));
    },
    [],
  );

  const refreshWorkspaces = useCallback(async () => {
    const result = await listWorkspaces();
    // 后端暂不可达时保留旧列表，下一轮重试
    if (result.ok) {
      const data = result.data;
      setWorkspaces(data);
    }
  }, []);

  const refreshVoiceprints = useCallback(async () => {
    const result = await listVoiceprints();
    // 后端暂不可达时保留旧列表，下一轮重试
    if (result.ok) {
      const data = result.data;
      setVoiceprints(data);
    }
  }, []);

  const refresh = useCallback(async () => {
    const result = await listTasks();
    // 后端暂不可达时保留旧列表，下一轮重试
    if (result.ok) {
      const data = result.data;
      setTasks(data.tasks);
      data.tasks.forEach(loadTimeline);
      data.tasks.forEach(maybeFetchTranscripts);
      data.tasks.forEach(syncCheckState);
    }
    refreshWorkspaces();
    refreshVoiceprints();
  }, [loadTimeline, maybeFetchTranscripts, syncCheckState, refreshWorkspaces, refreshVoiceprints]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [refresh]);

  return {
    tasks,
    transcripts,
    workspaces,
    voiceprints,
    refresh,
    refreshWorkspaces,
    refreshVoiceprints,
  };
}
