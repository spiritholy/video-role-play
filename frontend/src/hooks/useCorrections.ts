// 人工校正 (Manual Correction) 的段级写操作：改派、时段切分、时段删除，
// 以及放弃全部校正。错误文案按任务键控，供时间轴区块就近展示。
import { useCallback, useState } from "react";
import { errorMessage } from "../api/client";
import {
  deleteCorrections,
  putSegmentSpeaker,
  putSegmentSplit,
  removeSegment,
} from "../api/corrections";
import { workspaceKey } from "../lib/workspace";
import type {
  MutationOutcome,
  SplitOutcome,
  SplitResult,
  Task,
  TimelineSegment,
  TimelineUpdate,
} from "../types";

export interface Corrections {
  correctionError: Record<string, string | null>;
  reassignSegment: (
    task: Task,
    segment: TimelineSegment,
    target: string,
  ) => Promise<boolean>;
  splitSegment: (
    task: Task,
    segment: TimelineSegment,
    results: SplitResult[],
  ) => Promise<SplitOutcome>;
  deleteSegment: (
    task: Task,
    segment: TimelineSegment,
  ) => Promise<MutationOutcome>;
  discardCorrections: (task: Task) => Promise<void>;
}

export function useCorrections(deps: {
  applyTimeline: (data: TimelineUpdate) => void;
}): Corrections {
  const { applyTimeline } = deps;
  const [correctionError, setCorrectionError] = useState<
    Record<string, string | null>
  >({});

  const reassignSegment = useCallback(
    async (task: Task, segment: TimelineSegment, target: string) => {
      setCorrectionError((prev) => ({ ...prev, [task.id]: null }));
      const result = await putSegmentSpeaker(task.id, segment.index, target);
      if (!result.ok) {
        const message = errorMessage(result, "改派说话人失败");
        setCorrectionError((prev) => ({ ...prev, [task.id]: message }));
        return false;
      }
      applyTimeline({
        taskId: task.id,
        workspaceKey: workspaceKey(task),
        timeline: result.data,
      });
      return true;
    },
    [applyTimeline],
  );

  const splitSegment = useCallback(
    async (
      task: Task,
      segment: TimelineSegment,
      results: SplitResult[],
    ): Promise<SplitOutcome> => {
      const result = await putSegmentSplit(task.id, segment.index, results);
      if (!result.ok) {
        return { ok: false, detail: errorMessage(result, "保存切分失败") };
      }
      applyTimeline({
        taskId: task.id,
        workspaceKey: workspaceKey(task),
        timeline: result.data,
      });
      return { ok: true, detail: null, timeline: result.data };
    },
    [applyTimeline],
  );

  const deleteSegment = useCallback(
    async (
      task: Task,
      segment: TimelineSegment,
    ): Promise<MutationOutcome> => {
      setCorrectionError((prev) => ({ ...prev, [task.id]: null }));
      const result = await removeSegment(task.id, segment.index);
      if (!result.ok) {
        const detail = errorMessage(result, "删除时段失败");
        setCorrectionError((prev) => ({ ...prev, [task.id]: detail }));
        return { ok: false, detail };
      }
      applyTimeline({
        taskId: task.id,
        workspaceKey: workspaceKey(task),
        timeline: result.data,
      });
      return { ok: true, detail: null };
    },
    [applyTimeline],
  );

  const discardCorrections = useCallback(
    async (task: Task) => {
      setCorrectionError((prev) => ({ ...prev, [task.id]: null }));
      const result = await deleteCorrections(task.id);
      if (!result.ok) {
        const message = errorMessage(result, "放弃校正失败");
        setCorrectionError((prev) => ({ ...prev, [task.id]: message }));
        return;
      }
      applyTimeline({
        taskId: task.id,
        workspaceKey: workspaceKey(task),
        timeline: result.data,
      });
    },
    [applyTimeline],
  );

  return {
    correctionError,
    reassignSegment,
    splitSegment,
    deleteSegment,
    discardCorrections,
  };
}
