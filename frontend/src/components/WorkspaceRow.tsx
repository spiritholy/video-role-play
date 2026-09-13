// 单个工作区行：占用大小、可复用标记与继续/进入/删除操作。
import type { WorkspaceItem } from "../types";

export interface WorkspaceRowProps {
  workspace: WorkspaceItem;
  hasCompletedTask: boolean;
  onResume: (workspace: WorkspaceItem) => Promise<void>;
  onOpen: (workspace: WorkspaceItem) => Promise<void>;
  onDelete: (workspace: WorkspaceItem) => Promise<void>;
}

export function WorkspaceRow({
  workspace,
  hasCompletedTask,
  onResume,
  onOpen,
  onDelete,
}: WorkspaceRowProps) {
  return (
    <div key={workspace.name} className="workspace-row">
      <div className="workspace-main">
        <span className="workspace-name">{workspace.name}</span>
        <span className="workspace-size">{workspace.size_label}</span>
        <span className={`workspace-tag${workspace.reusable ? " tag-reusable" : ""}`}>
          {workspace.reusable ? "分析可复用" : "分析未完成"}
        </span>
        {workspace.busy && <span className="workspace-tag tag-busy">任务使用中</span>}
        {workspace.resumable && (
          <button
            className="resume"
            disabled={workspace.busy}
            title={
              workspace.busy
                ? "任务正在使用该工作区，取消任务后再继续"
                : "从断点继续处理该源视频，不重算已完成阶段"
            }
            onClick={() => onResume(workspace)}
          >
            继续
          </button>
        )}
        {workspace.openable && !hasCompletedTask && (
          <button
            className="resume"
            disabled={workspace.busy}
            title={
              workspace.busy
                ? "任务正在使用该工作区，稍后再进入"
                : "复用已完成的核心分析，进入擦除目标说话人选择（认人/转写会快速补跑）"
            }
            onClick={() => onOpen(workspace)}
          >
            选择屏蔽人声
          </button>
        )}
        <button
          className="delete"
          disabled={workspace.busy}
          title={
            workspace.busy
              ? "任务正在使用该工作区，取消任务后再删除"
              : `删除工作区，释放 ${workspace.size_label}`
          }
          onClick={() => onDelete(workspace)}
        >
          删除
        </button>
      </div>
      <div className="workspace-path">{workspace.source_path ?? workspace.path}</div>
    </div>
  );
}
