// 工作区区块：占用统计、声纹库面板与工作区操作列表。
import { VoiceprintLibraryPanel } from "./VoiceprintLibraryPanel";
import { WorkspaceRow } from "./WorkspaceRow";
import type {
  Task,
  VoiceprintList,
  WorkspaceItem,
  WorkspaceList,
} from "../types";

export interface WorkspacePanelProps {
  workspaces: WorkspaceList | null;
  workspaceError: string | null;
  voiceprints: VoiceprintList | null;
  voiceprintError: string | null;
  tasks: Task[];
  handleResumeWorkspace: (w: WorkspaceItem) => Promise<void>;
  handleOpenWorkspace: (w: WorkspaceItem) => Promise<void>;
  handleDeleteWorkspace: (w: WorkspaceItem) => Promise<void>;
  onDeleteVoiceprint: (name: string) => Promise<void>;
}

export function WorkspacePanel({
  workspaces,
  workspaceError,
  voiceprints,
  voiceprintError,
  tasks,
  handleResumeWorkspace,
  handleOpenWorkspace,
  handleDeleteWorkspace,
  onDeleteVoiceprint,
}: WorkspacePanelProps) {
  return (
    <section className="workspaces">
      <h2>工作区{workspaces ? `（共占用 ${workspaces.total_label}）` : ""}</h2>
      {workspaceError && <p className="error-text">{workspaceError}</p>}
      <VoiceprintLibraryPanel
        voiceprints={voiceprints}
        voiceprintError={voiceprintError}
        onDeleteVoiceprint={onDeleteVoiceprint}
      />
      {workspaces && workspaces.workspaces.length === 0 && (
        <p className="empty">
          还没有工作区。任务开始运行后，中间产物会写入工作区目录，可在此手动删除释放磁盘。
        </p>
      )}
      {workspaces?.workspaces.map((w) => (
        <WorkspaceRow
          key={w.name}
          workspace={w}
          hasCompletedTask={tasks.some(
            (t) => t.workspace_dir === w.path && t.state === "completed",
          )}
          onResume={handleResumeWorkspace}
          onOpen={handleOpenWorkspace}
          onDelete={handleDeleteWorkspace}
        />
      ))}
    </section>
  );
}
