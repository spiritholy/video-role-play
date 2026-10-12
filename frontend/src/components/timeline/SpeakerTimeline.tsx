// 说话人时间轴总览：每人一行（名字、时长、存入声纹库、时段轨道、样本台词），
// 并挂载时段校正详情弹窗。
import { useState } from "react";
import { fmtTime } from "../../lib/format";
import { firstSegment } from "../../lib/segments";
import { shownName, speakerColor } from "../../lib/speakers";
import { DetailModal } from "../detail/DetailModal";
import { SegmentTrack } from "./SegmentTrack";
import { SpeakerName } from "./SpeakerName";
import { SpeakerSamples } from "./SpeakerSamples";
import type {
  Playing,
  SplitResult,
  Timeline,
  TimelineSegment,
  TranscriptSample,
} from "../../types";

export interface SpeakerTimelineProps {
  taskId: string;
  timeline: Timeline;
  names: Record<string, string>;
  samplesBySpeaker: Record<string, TranscriptSample[]>;
  samplesHint: string | null;
  playing: Playing | null;
  playPos: number | null;
  onPlay: (
    taskId: string,
    speaker: string,
    seg: Pick<TimelineSegment, "start" | "end">,
  ) => void;
  onDelete: (
    segment: TimelineSegment,
  ) => Promise<{ ok: boolean; detail: string | null }>;
  onRename: (label: string, name: string) => Promise<string | null | undefined>;
  onStoreVoiceprint: (label: string) => Promise<void>;
  voiceprintBusy: (label: string) => boolean;
  onReassign: (segment: TimelineSegment, target: string) => Promise<boolean>;
  onSplit: (
    segment: TimelineSegment,
    results: SplitResult[],
  ) => Promise<{ ok: boolean; detail: string | null; timeline?: Timeline }>;
  onDiscard: () => Promise<void>;
  correctionError: string | null;
}

export function SpeakerTimeline({
  taskId,
  timeline,
  names,
  samplesBySpeaker,
  samplesHint,
  playing,
  playPos,
  onPlay,
  onDelete,
  onRename,
  onStoreVoiceprint,
  voiceprintBusy,
  onReassign,
  onSplit,
  onDiscard,
  correctionError,
}: SpeakerTimelineProps) {
  const [detail, setDetail] = useState<{
    speaker: string;
    segment: TimelineSegment;
  } | null>(null);
  const correction = timeline.correction;

  function openDetail(speaker?: string, segment?: TimelineSegment) {
    if (speaker && segment) {
      setDetail({ speaker, segment });
      return;
    }
    const first = firstSegment(timeline);
    if (first) setDetail(first);
  }

  return (
    <div className="timeline">
      <div className="timeline-title">
        说话人时间轴总览（共 {timeline.num_speakers} 人，点击总览打开校正详情；点击名字可命名）
      </div>
      {samplesHint && <p className="timeline-hint">{samplesHint}</p>}
      {correction.message && <p className="correction-warning">{correction.message}</p>}
      {correction.locked_reason && correction.state !== "invalidated" && (
        <p className="correction-warning">{correction.locked_reason}</p>
      )}
      {(correction.state === "active" || correction.state === "invalidated") && (
        <div className="correction-actions">
          <button
            type="button"
            onClick={() => onDiscard()}
            disabled={correction.state !== "invalidated" && !correction.editable}
          >
            {correction.state === "invalidated" ? "放弃作废校正" : "放弃全部校正"}
          </button>
        </div>
      )}
      {correctionError && <p className="error-text">{correctionError}</p>}
      <div className="timeline-canvas">
        {timeline.speakers.map((sp) => {
          const color = speakerColor(timeline.speakers, sp.label);
          const samples = samplesBySpeaker[sp.label] ?? [];
          return (
            <div key={sp.label} className="timeline-row">
              <span className="speaker-chip">
                <span className="dot" style={{ background: color }} />
                <SpeakerName
                  label={sp.label}
                  name={names[sp.label] ?? sp.name ?? null}
                  auto={Boolean(sp.auto && !names[sp.label])}
                  editable={correction.editable}
                  onRename={onRename}
                />
                <span className="speech-time">{fmtTime(sp.speech_time)}</span>
                <button
                  type="button"
                  className="store-voiceprint"
                  disabled={voiceprintBusy(sp.label)}
                  title={`从 ${shownName(names, sp)} 的当前生效时段抽取声纹并存入声纹库`}
                  onClick={() => void onStoreVoiceprint(sp.label)}
                >
                  {voiceprintBusy(sp.label) ? "存入中…" : "存入声纹库"}
                </button>
              </span>
              <SegmentTrack
                taskId={taskId}
                duration={timeline.duration}
                speaker={sp}
                color={color}
                playing={playing}
                selectedKey={null}
                onSelect={(speakerLabel, seg) => openDetail(speakerLabel, seg)}
                onBlankClick={() => openDetail()}
                variant="main"
              />
              <SpeakerSamples
                taskId={taskId}
                speaker={sp}
                names={names}
                samples={samples}
                playing={playing}
                onPlay={onPlay}
              />
            </div>
          );
        })}
        {timeline.duration > 0 && (
          <div className="timeline-axis">
            <span />
            <div className="timeline-axis-canvas">
              {Array.from({ length: 5 }, (_, i) => (
                <span key={i}>{fmtTime((timeline.duration * i) / 4)}</span>
              ))}
            </div>
          </div>
        )}
      </div>
      {detail && (
        <DetailModal
          taskId={taskId}
          timeline={timeline}
          names={names}
          playing={playing}
          playPos={playPos}
          editable={correction.editable}
          onPlay={onPlay}
          onDelete={onDelete}
          onReassign={onReassign}
          onSplit={onSplit}
          initial={detail}
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  );
}
