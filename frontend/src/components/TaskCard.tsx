// 单个处理任务卡片：状态头、阶段进度、分轨试听、声纹校验与说话人时间轴。
import { fileName, stageSummary } from "../lib/format";
import { StageChips } from "./StageChips";
import { StemPlayers } from "./StemPlayers";
import { TaskTimelineSection } from "./TaskTimelineSection";
import { VoiceprintCheckPanel } from "./VoiceprintCheckPanel";
import type {
  CheckActions,
  CorrectionActions,
  NamingActions,
  PlaybackApi,
  Task,
  TimelineOrError,
  TranscriptsOrError,
  VoiceprintCheckState,
} from "../types";

export interface TaskCardProps {
  task: Task;
  timelines: Record<string, TimelineOrError>;
  speakerNames: Record<string, Record<string, string>>;
  transcripts: Record<string, TranscriptsOrError>;
  ingestWarnings: Record<string, string | null>;
  correctionError: Record<string, string | null>;
  checkStates: Record<string, VoiceprintCheckState>;
  onCancel: (id: string) => Promise<void>;
  onResume: (id: string) => Promise<void>;
  corrections: CorrectionActions;
  naming: NamingActions;
  checks: CheckActions;
  playback: PlaybackApi;
}

export function TaskCard({
  task,
  timelines,
  speakerNames,
  transcripts,
  ingestWarnings,
  correctionError,
  checkStates,
  onCancel,
  onResume,
  corrections,
  naming,
  checks,
  playback,
}: TaskCardProps) {
const cancellable = task.state === "queued" || task.state === "running";
const resumable = task.state === "failed" || task.state === "cancelled";
  return (
    <div key={task.id} className={`task-card state-${task.state}`}>
      <div className="task-head">
        <span className="task-name">{fileName(task.video_path)}</span>
        <span className={`badge badge-${task.state}`}>{task.state_label}</span>
        {cancellable && (
          <button className="cancel" onClick={() => onCancel(task.id)}>
            取消
          </button>
        )}
        {resumable && (
          <button className="resume" onClick={() => onResume(task.id)}>
            继续
          </button>
        )}
      </div>
      <div className="task-meta">
        <span>{task.video_path}</span>
        {task.audio_track != null && <span>音轨 #{task.audio_track + 1}</span>}
      </div>
      <div className="task-stages">
        <span>{stageSummary(task)}</span>
        <StageChips stages={task.stages} />
      </div>
      {resumable && (
        <p className="task-hint">
          已完成阶段的产物保留在工作区，点“继续”从断点阶段跑起，不重算分离与说话人分离。
        </p>
      )}
      <StemPlayers taskId={task.id} stages={task.stages} />
      {task.stages.some((s) => s.id === "diarize" && s.status === "done") && (
        <VoiceprintCheckPanel task={task} checkStates={checkStates} checks={checks} />
      )}
      {task.stages.some((s) => s.id === "diarize" && s.status === "done") && (
        <TaskTimelineSection
          task={task}
          timelines={timelines}
          transcripts={transcripts}
          speakerNames={speakerNames}
          ingestWarnings={ingestWarnings}
          correctionError={correctionError}
          corrections={corrections}
          naming={naming}
          playback={playback}
        />
      )}
      {task.error && <p className="error-text">失败原因：{task.error}</p>}
      {task.output_path && (
        <p className="output-text">成品：{task.output_path}</p>
      )}
    </div>
  );
}
