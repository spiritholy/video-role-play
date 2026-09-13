// 人工校正 (Manual Correction) 的写接口：改派、时段切分、时段删除、放弃全部
// 校正，以及说话人命名。成功时一律返回校正后的最新时间轴。
import { request, jsonInit, type ApiResult } from "./client";
import type { SplitResult, Timeline } from "../types";

export interface SpeakerNameResult {
  label: string;
  name: string | null;
}

export function putSegmentSpeaker(
  taskId: string,
  index: number,
  target: string,
): Promise<ApiResult<Timeline>> {
  return request<Timeline>(
    `/api/tasks/${taskId}/segments/${index}/speaker`,
    jsonInit("PUT", { target }),
  );
}

export function putSegmentSplit(
  taskId: string,
  index: number,
  results: SplitResult[],
): Promise<ApiResult<Timeline>> {
  return request<Timeline>(
    `/api/tasks/${taskId}/segments/${index}/split`,
    jsonInit("PUT", { results }),
  );
}

export function removeSegment(
  taskId: string,
  index: number,
): Promise<ApiResult<Timeline>> {
  return request<Timeline>(`/api/tasks/${taskId}/segments/${index}`, {
    method: "DELETE",
  });
}

export function deleteCorrections(taskId: string): Promise<ApiResult<Timeline>> {
  return request<Timeline>(`/api/tasks/${taskId}/corrections`, {
    method: "DELETE",
  });
}

export function putSpeakerName(
  taskId: string,
  label: string,
  name: string,
): Promise<ApiResult<SpeakerNameResult>> {
  return request<SpeakerNameResult>(
    `/api/tasks/${taskId}/speakers/${encodeURIComponent(label)}/name`,
    jsonInit("PUT", { name }),
  );
}
