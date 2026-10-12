// 任务卡内的时间轴区块：加载态/错误态、样本台词派生、校正入口与擦除表单。
import { deriveSamples } from "../lib/samples";
import { workspaceKey } from "../lib/workspace";
import { EraseForm } from "./EraseForm";
import { SpeakerTimeline } from "./timeline/SpeakerTimeline";
import type {
  CorrectionActions,
  NamingActions,
  PlaybackApi,
  Task,
  TimelineOrError,
  TranscriptsOrError,
} from "../types";

export interface TaskTimelineSectionProps {
  task: Task;
  timelines: Record<string, TimelineOrError>;
  transcripts: Record<string, TranscriptsOrError>;
  speakerNames: Record<string, Record<string, string>>;
  ingestWarnings: Record<string, string | null>;
  correctionError: Record<string, string | null>;
  corrections: CorrectionActions;
  naming: NamingActions;
  playback: PlaybackApi;
}

export function TaskTimelineSection({
  task,
  timelines,
  transcripts,
  speakerNames,
  ingestWarnings,
  correctionError,
  corrections,
  naming,
  playback,
}: TaskTimelineSectionProps) {
  const timeline = timelines[task.id];
  if (timeline === undefined) {
    return <p className="timeline-hint">正在加载说话人时间轴…</p>;
  }
  if (timeline === "error") {
    return <p className="timeline-hint">说话人时间轴加载失败</p>;
  }

  const transcribeStage = task.stages.find((s) => s.id === "transcribe");
  const { samplesHint, samplesBySpeaker } = deriveSamples(
    transcripts[task.id],
    transcribeStage,
  );

  const ingestWarning = ingestWarnings[task.id];
  return (
    <>
      <SpeakerTimeline
        taskId={task.id}
        timeline={timeline}
        names={speakerNames[workspaceKey(task)] ?? {}}
        samplesBySpeaker={samplesBySpeaker}
        samplesHint={samplesHint}
        playing={playback.playing}
        playPos={playback.playPos}
        onPlay={playback.playSegment}
        onDelete={(segment) => corrections.deleteSegment(task, segment)}
        onRename={(label, name) => naming.renameSpeaker(task, label, name)}
        onStoreVoiceprint={(label) => naming.storeVoiceprint(task, label)}
        voiceprintBusy={(rowLabel) => naming.isVoiceprintBusy(task, rowLabel)}
        onReassign={(segment, target) =>
          corrections.reassignSegment(task, segment, target)
        }
        onSplit={(segment, results) =>
          corrections.splitSegment(task, segment, results)
        }
        onDiscard={() => corrections.discardCorrections(task)}
        correctionError={correctionError[task.id] ?? null}
      />
      {ingestWarning && <p className="error-text">{ingestWarning}</p>}
      {task.state === "completed" && (
        <EraseForm
          task={task}
          speakers={timeline.speakers}
          names={speakerNames[workspaceKey(task)] ?? {}}
        />
      )}
    </>
  );
}
