// 处理任务 (Task) 的列表、发起、取消、续跑，与音轨探测、发起擦除。
import { request, jsonInit, type ApiResult } from "./client";
import type { ProbeResult, Task } from "../types";

export interface TaskListResponse {
  tasks: Task[];
}

export function listTasks(): Promise<ApiResult<TaskListResponse>> {
  return request<TaskListResponse>("/api/tasks");
}

export function createTask(body: {
  video_path: string;
  audio_track: number | null;
  num_speakers: number | null;
}): Promise<ApiResult<unknown>> {
  return request<unknown>("/api/tasks", jsonInit("POST", body));
}

export function cancelTask(id: string): Promise<ApiResult<unknown>> {
  return request<unknown>(`/api/tasks/${id}/cancel`, { method: "POST" });
}

export function resumeTask(id: string): Promise<ApiResult<unknown>> {
  return request<unknown>(`/api/tasks/${id}/resume`, { method: "POST" });
}

export function probeVideo(videoPath: string): Promise<ApiResult<ProbeResult>> {
  return request<ProbeResult>("/api/probe", jsonInit("POST", { video_path: videoPath }));
}

export function startErase(
  taskId: string,
  speakers: string[],
  outputDir: string | null,
): Promise<ApiResult<unknown>> {
  return request<unknown>(
    `/api/tasks/${taskId}/erase`,
    jsonInit("POST", { speakers, output_dir: outputDir }),
  );
}
