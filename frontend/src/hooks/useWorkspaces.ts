// 工作区 (Workspace) 的三个动作：从断点续跑、复用分析产物进入擦除界面、删除释放磁盘。
import { useCallback, useState } from "react";
import { errorMessage } from "../api/client";
import {
  deleteWorkspace,
  openWorkspace,
  resumeWorkspace,
} from "../api/workspaces";
import type { WorkspaceItem } from "../types";

export interface WorkspaceActions {
  workspaceError: string | null;
  handleResumeWorkspace: (w: WorkspaceItem) => Promise<void>;
  handleOpenWorkspace: (w: WorkspaceItem) => Promise<void>;
  handleDeleteWorkspace: (w: WorkspaceItem) => Promise<void>;
}

export function useWorkspaces(deps: {
  refresh: () => Promise<void>;
}): WorkspaceActions {
  const { refresh } = deps;
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  const handleResumeWorkspace = useCallback(
    async (w: WorkspaceItem) => {
      setWorkspaceError(null);
      const result = await resumeWorkspace(w.name);
      if (!result.ok) {
        setWorkspaceError(errorMessage(result, "续跑失败"));
        return;
      }
      refresh();
    },
    [refresh],
  );

  const handleOpenWorkspace = useCallback(
    async (w: WorkspaceItem) => {
      setWorkspaceError(null);
      const result = await openWorkspace(w.name);
      if (!result.ok) {
        setWorkspaceError(errorMessage(result, "打开擦除界面失败"));
        return;
      }
      refresh();
    },
    [refresh],
  );

  const handleDeleteWorkspace = useCallback(
    async (w: WorkspaceItem) => {
      const confirmed = window.confirm(
        `删除工作区「${w.name}」？\n\n释放 ${w.size_label}，` +
          `全部中间产物一并删除。\n再次处理该源视频需要从头分析（数小时）。`,
      );
      if (!confirmed) return;
      setWorkspaceError(null);
      const result = await deleteWorkspace(w.name);
      if (!result.ok) {
        setWorkspaceError(errorMessage(result, "删除工作区失败"));
        return;
      }
      refresh();
    },
    [refresh],
  );

  return {
    workspaceError,
    handleResumeWorkspace,
    handleOpenWorkspace,
    handleDeleteWorkspace,
  };
}
