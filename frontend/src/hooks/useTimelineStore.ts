// 时间轴与说话人命名的前端镜像：timelines 按任务键控，speakerNames 按工作区键控。
// 写操作成功后由 applyTimeline 统一回灌，保证「生效时段 = 分离结果 + 人工校正」。
import { useCallback, useState } from "react";
import { fetchDiarization } from "../api/analysis";
import { workspaceKey } from "../lib/workspace";
import type {
  Task,
  Timeline,
  TimelineOrError,
  TimelineUpdate,
} from "../types";

export interface TimelineStore {
  timelines: Record<string, TimelineOrError>;
  speakerNames: Record<string, Record<string, string>>;
  loadTimeline: (t: Task) => Promise<void>;
  applyTimeline: (data: TimelineUpdate) => void;
  applySpeakerName: (
    wsKey: string,
    label: string,
    name: string | null,
  ) => void;
}

export function useTimelineStore(): TimelineStore {
  const [timelines, setTimelines] = useState<Record<string, TimelineOrError>>({});
  // 说话人命名（工作区元数据的本地镜像）：工作区 → 编号标签 → 名字。
  // 按工作区键控：命名是工作区级数据，同源多任务卡片共享同一份显示。
  const [speakerNames, setSpeakerNames] = useState<
    Record<string, Record<string, string>>
  >({});

  // 本地镜像只收手工命名：自动认出来的名字带 auto 标记，由时间轴接口
  // 的 name/auto 直接展示，不冒充手工命名（否则「自动」徽标会丢失）
  const mirrorNames = useCallback((wsKey: string, timeline: Timeline) => {
    setSpeakerNames((prev) => ({
      ...prev,
      [wsKey]: Object.fromEntries(
        timeline.speakers
          .filter((s) => s.name && !s.auto)
          .map((s) => [s.label, s.name as string]),
      ),
    }));
  }, []);

  const loadTimeline = useCallback(
    async (t: Task) => {
      if (!t.stages.some((s) => s.id === "diarize" && s.status === "done")) return;
      const result = await fetchDiarization(t.id);
      if (!result.ok) {
        setTimelines((prev) => ({ ...prev, [t.id]: "error" }));
        return;
      }
      const data = result.data;
      setTimelines((prev) => ({ ...prev, [t.id]: data }));
      mirrorNames(workspaceKey(t), data);
    },
    [mirrorNames],
  );

  const applyTimeline = useCallback(
    (data: TimelineUpdate) => {
      setTimelines((prev) => ({ ...prev, [data.taskId]: data.timeline }));
      mirrorNames(data.workspaceKey, data.timeline);
    },
    [mirrorNames],
  );

  const applySpeakerName = useCallback(
    (wsKey: string, label: string, name: string | null) => {
      setSpeakerNames((prev) => {
        const forWs = { ...(prev[wsKey] ?? {}) };
        if (name) forWs[label] = name;
        else delete forWs[label];
        return { ...prev, [wsKey]: forWs };
      });
    },
    [],
  );

  return {
    timelines,
    speakerNames,
    loadTimeline,
    applyTimeline,
    applySpeakerName,
  };
}
