// 处理任务 (Task) 的取消与续跑：续跑失败的原因显示在任务列表顶部。
import { useCallback, useState } from "react";
import { errorMessage } from "../api/client";
import { cancelTask, resumeTask } from "../api/tasks";

export interface TaskActions {
  resumeError: string | null;
  handleCancel: (id: string) => Promise<void>;
  handleResume: (id: string) => Promise<void>;
}

export function useTaskActions(deps: {
  refresh: () => Promise<void>;
}): TaskActions {
  const { refresh } = deps;
  const [resumeError, setResumeError] = useState<string | null>(null);

  const handleCancel = useCallback(
    async (id: string) => {
      await cancelTask(id);
      refresh();
    },
    [refresh],
  );

  const handleResume = useCallback(
    async (id: string) => {
      setResumeError(null);
      const result = await resumeTask(id);
      if (!result.ok) {
        setResumeError(errorMessage(result, "续跑失败"));
        return;
      }
      refresh();
    },
    [refresh],
  );

  return { resumeError, handleCancel, handleResume };
}
