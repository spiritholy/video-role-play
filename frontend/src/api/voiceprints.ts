// 声纹库 (Voiceprint Library) 与声纹校验 (Voiceprint Check) 的接口。
import { request, jsonInit, type ApiResult } from "./client";
import type {
  Timeline,
  VoiceprintCheckResult,
  VoiceprintCheckState,
  VoiceprintList,
} from "../types";

export interface StoreVoiceprintResponse {
  voiceprint_error?: string | null;
}

export interface StartCheckResponse {
  voiceprint_check?: VoiceprintCheckState | null;
}

export interface SuggestionDecisionResponse {
  result: VoiceprintCheckResult;
  timeline: Timeline;
}

export function listVoiceprints(): Promise<ApiResult<VoiceprintList>> {
  return request<VoiceprintList>("/api/voiceprints");
}

export function deleteVoiceprintEntry(
  name: string,
): Promise<ApiResult<unknown>> {
  return request<unknown>(`/api/voiceprints/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function storeSpeakerVoiceprint(
  taskId: string,
  label: string,
): Promise<ApiResult<StoreVoiceprintResponse>> {
  return request<StoreVoiceprintResponse>(
    `/api/tasks/${taskId}/speakers/${encodeURIComponent(label)}/voiceprint`,
    { method: "POST" },
  );
}

export function getVoiceprintCheck(
  taskId: string,
): Promise<ApiResult<VoiceprintCheckState>> {
  return request<VoiceprintCheckState>(`/api/tasks/${taskId}/voiceprint-check`);
}

export function postVoiceprintCheck(
  taskId: string,
): Promise<ApiResult<StartCheckResponse>> {
  return request<StartCheckResponse>(`/api/tasks/${taskId}/voiceprint-check`, {
    method: "POST",
  });
}

export function deleteVoiceprintCheck(
  taskId: string,
): Promise<ApiResult<unknown>> {
  return request<unknown>(`/api/tasks/${taskId}/voiceprint-check`, {
    method: "DELETE",
  });
}

export function postSuggestionDecision(
  taskId: string,
  index: number,
  decision: "accept" | "ignore",
): Promise<ApiResult<SuggestionDecisionResponse>> {
  return request<SuggestionDecisionResponse>(
    `/api/tasks/${taskId}/voiceprint-check/suggestions/${index}/decision`,
    jsonInit("POST", { decision }),
  );
}
