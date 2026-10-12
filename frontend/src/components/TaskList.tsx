// 处理任务列表区块。
import { TaskCard } from "./TaskCard";
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

export interface TaskListProps {
  tasks: Task[];
  resumeError: string | null;
  checkError: string | null;
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

export function TaskList({
  tasks,
  resumeError,
  checkError,
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
}: TaskListProps) {
  return (
    <section className="tasks">
      <h2>处理任务</h2>
      {resumeError && <p className="error-text">{resumeError}</p>}
      {checkError && <p className="error-text">{checkError}</p>}
      {tasks.length === 0 && <p className="empty">还没有任务，提交一个源视频开始。</p>}
      {tasks.map((t) => (
        <TaskCard
          key={t.id}
          task={t}
          timelines={timelines}
          speakerNames={speakerNames}
          transcripts={transcripts}
          ingestWarnings={ingestWarnings}
          correctionError={correctionError}
          checkStates={checkStates}
          onCancel={onCancel}
          onResume={onResume}
          corrections={corrections}
          naming={naming}
          checks={checks}
          playback={playback}
        />
      ))}
    </section>
  );
}
