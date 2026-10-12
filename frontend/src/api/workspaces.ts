// 工作区 (Workspace) 的列表、续跑、进入擦除界面与删除。
import { request, type ApiResult } from "./client";
import type { WorkspaceList } from "../types";

export function listWorkspaces(): Promise<ApiResult<WorkspaceList>> {
  return request<WorkspaceList>("/api/workspaces");
}

export function resumeWorkspace(name: string): Promise<ApiResult<unknown>> {
  return request<unknown>(`/api/workspaces/${encodeURIComponent(name)}/resume`, {
    method: "POST",
  });
}

export function openWorkspace(name: string): Promise<ApiResult<unknown>> {
  return request<unknown>(`/api/workspaces/${encodeURIComponent(name)}/open`, {
    method: "POST",
  });
}

export function deleteWorkspace(name: string): Promise<ApiResult<unknown>> {
  return request<unknown>(`/api/workspaces/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}
