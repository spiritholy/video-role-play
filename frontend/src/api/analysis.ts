// 说话人分离时间轴与样本段转写的读取接口。
import { request, type ApiResult } from "./client";
import type { Timeline, Transcripts } from "../types";

export function fetchDiarization(taskId: string): Promise<ApiResult<Timeline>> {
  return request<Timeline>(`/api/tasks/${taskId}/diarization`);
}

export function fetchTranscripts(taskId: string): Promise<ApiResult<Transcripts>> {
  return request<Transcripts>(`/api/tasks/${taskId}/transcripts`);
}
