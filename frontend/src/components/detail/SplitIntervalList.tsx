// 切分结果区间列表：逐个区间试听与分配说话人（现有说话人或草稿槽位）。
import { fmtTime } from "../../lib/format";
import { shownName } from "../../lib/speakers";
import type { SplitDraft } from "../../hooks/useSplitDraft";
import type { SplitInterval, Timeline, TimelineSegment } from "../../types";

export interface SplitIntervalListProps {
  draft: SplitDraft;
  timeline: Timeline;
  names: Record<string, string>;
  editable: boolean;
  onPlay: (seg: Pick<TimelineSegment, "start" | "end">) => void;
  isIntervalPlaying: (iv: SplitInterval) => boolean;
}

export function SplitIntervalList({
  draft,
  timeline,
  names,
  editable,
  onPlay,
  isIntervalPlaying,
}: SplitIntervalListProps) {
  const { intervals, targets, slots, setTargets, setError } = draft;
  return (
<div className="split-intervals">
  <div className="split-intervals-title">
    结果区间（{intervals.length} 个）——默认继承原说话人，可改选现有说话人或新建槽位
  </div>
  {intervals.map((iv, i) => (
    <div
      key={iv.key}
      className={`split-interval-row${isIntervalPlaying(iv) ? " playing" : ""}`}
    >
      <span className="split-interval-time">
        {fmtTime(iv.start)} – {fmtTime(iv.end)}
      </span>
      <button
        type="button"
        className="split-interval-play"
        onClick={() => onPlay({ start: iv.start, end: iv.end })}
      >
        {isIntervalPlaying(iv) ? "试听中" : "试听"}
      </button>
      <select
        value={targets[i] ?? ""}
        disabled={!editable}
        onChange={(e) => {
          const next = [...targets];
          next[i] = e.target.value;
          setTargets(next);
          setError(null);
        }}
      >
        <option value="" disabled>
          选择说话人
        </option>
        {timeline.speakers.map((sp) => (
          <option key={sp.label} value={sp.label}>
            {shownName(names, sp)}
          </option>
        ))}
        {slots.map((slot, n) => (
          <option key={slot} value={slot}>
            新建说话人 {n + 1}
          </option>
        ))}
      </select>
    </div>
  ))}
</div>
  );
}
