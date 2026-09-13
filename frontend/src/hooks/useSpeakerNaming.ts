// 说话人命名 (Speaker Name) 与显式存入声纹库：命名是工作区级数据，存库取的是
// 该说话人的当前生效时段。入库失败的后端警告按任务键控，仅前端会话态。
import { useCallback, useState } from "react";
import { errorMessage } from "../api/client";
import { putSpeakerName } from "../api/corrections";
import { storeSpeakerVoiceprint } from "../api/voiceprints";
import { workspaceKey } from "../lib/workspace";
import type { Task, TimelineOrError } from "../types";

export interface SpeakerNaming {
  // 命名入库失败警告（任务 → 文案）：仅前端会话态，下次命名成功或刷新即清除。
  ingestWarnings: Record<string, string | null>;
  voiceprintBusy: Record<string, boolean>;
  renameSpeaker: (
    task: Task,
    label: string,
    rawName: string,
  ) => Promise<string | null | undefined>;
  handleStoreVoiceprint: (task: Task, label: string) => Promise<void>;
}

export function useSpeakerNaming(deps: {
  timelines: Record<string, TimelineOrError>;
  speakerNames: Record<string, Record<string, string>>;
  applySpeakerName: (
    wsKey: string,
    label: string,
    name: string | null,
  ) => void;
  refreshVoiceprints: () => Promise<void>;
}): SpeakerNaming {
  const { timelines, speakerNames, applySpeakerName, refreshVoiceprints } = deps;
  const [ingestWarnings, setIngestWarnings] = useState<Record<string, string | null>>({});
  const [voiceprintBusy, setVoiceprintBusy] = useState<Record<string, boolean>>({});

  const renameSpeaker = useCallback(
    async (
      task: Task,
      label: string,
      rawName: string,
    ): Promise<string | null | undefined> => {
      const wsKey = workspaceKey(task);
      const result = await putSpeakerName(task.id, label, rawName);
      if (!result.ok) return undefined;
      const data = result.data;
      setIngestWarnings((prev) => ({ ...prev, [task.id]: null }));
      applySpeakerName(wsKey, label, data.name);
      return data.name;
    },
    [applySpeakerName],
  );

  const handleStoreVoiceprint = useCallback(
    async (task: Task, label: string) => {
      const key = `${task.id}:${label}`;
      if (voiceprintBusy[key]) return;
      const timeline = timelines[task.id];
      const speaker =
        timeline === "error" || timeline === undefined
          ? undefined
          : timeline.speakers.find((sp) => sp.label === label);
      const name =
        speakerNames[workspaceKey(task)]?.[label] ?? speaker?.name ?? null;
      if (!name) {
        setIngestWarnings((prev) => ({
          ...prev,
          [task.id]: "请先命名说话人，再存入声纹库",
        }));
        return;
      }

      setVoiceprintBusy((prev) => ({ ...prev, [key]: true }));
      setIngestWarnings((prev) => ({ ...prev, [task.id]: null }));
      const result = await storeSpeakerVoiceprint(task.id, label);
      if (!result.ok) {
        const message = errorMessage(result, "存入声纹库失败");
        setIngestWarnings((prev) => ({ ...prev, [task.id]: message }));
      } else {
        const data = result.data;
        setIngestWarnings((prev) => ({
          ...prev,
          [task.id]: data?.voiceprint_error
            ? `声纹入库失败：${data.voiceprint_error}`
            : null,
        }));
      }
      setVoiceprintBusy((prev) => ({ ...prev, [key]: false }));
      refreshVoiceprints();
    },
    [refreshVoiceprints, speakerNames, timelines, voiceprintBusy],
  );

  return {
    ingestWarnings,
    voiceprintBusy,
    renameSpeaker,
    handleStoreVoiceprint,
  };
}
