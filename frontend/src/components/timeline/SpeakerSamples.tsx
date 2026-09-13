// 说话人的样本台词试听入口（转写样本段）。
import { fmtTime } from "../../lib/format";
import { isSegmentActive } from "../../lib/segments";
import { shownName } from "../../lib/speakers";
import type {
  Playing,
  TimelineSegment,
  TimelineSpeaker,
  TranscriptSample,
} from "../../types";

export interface SpeakerSamplesProps {
  taskId: string;
  speaker: TimelineSpeaker;
  names: Record<string, string>;
  samples: TranscriptSample[];
  playing: Playing | null;
  onPlay: (
    taskId: string,
    speaker: string,
    seg: Pick<TimelineSegment, "start" | "end">,
  ) => void;
}

export function SpeakerSamples({
  taskId,
  speaker,
  names,
  samples,
  playing,
  onPlay,
}: SpeakerSamplesProps) {
  if (samples.length === 0) return null;
  return (
    <div className="speaker-samples">
      <span className="samples-label">样本台词</span>
      <div className="sample-list">
        {samples.map((s, i) => (
          <button
            key={i}
            type="button"
            className={`sample${
              isSegmentActive(taskId, playing, speaker.label, s) ? " playing" : ""
            }`}
            title={`试听 ${shownName(names, speaker)} 的样本 ${fmtTime(s.start)} – ${fmtTime(s.end)}`}
            onClick={() => onPlay(taskId, speaker.label, s)}
          >
            <span className="sample-time">
              {fmtTime(s.start)}–{fmtTime(s.end)}
            </span>
            <span className="sample-text">
              {s.text || "（无可识别台词）"}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
