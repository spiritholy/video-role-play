// 工作区 (Workspace) 键：同源多任务共享同一份说话人命名。
import type { Task } from "../types";

export function workspaceKey(t: Task): string {
  return t.workspace_dir ?? t.id;
}
