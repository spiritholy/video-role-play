// 选中时段操作面板：试听、整段改派、新建槽位、删除时段与保存切分。
import { useState } from "react";
import { fmtTime } from "../../lib/format";
import { shownName } from "../../lib/speakers";
import {
  splitPlayBounds,
  toSplitResults,
  validateSplitDraft,
} from "../../lib/split";
import { useSplitDraft } from "../../hooks/useSplitDraft";
import { SplitIntervalList } from "./SplitIntervalList";
import { SplitTrack } from "./SplitTrack";
import type {
  Playing,
  SplitInterval,
  SplitResult,
  Timeline,
  TimelineSegment,
} from "../../types";

export interface SplitPanelProps {
  taskId: string;
  segment: TimelineSegment;
  originalSpeaker: string;
  speakerColorValue: string;
  timeline: Timeline;
  names: Record<string, string>;
  editable: boolean;
  playing: Playing | null;
  playPos: number | null;
  onPlay: (seg: Pick<TimelineSegment, "start" | "end">) => void;
  onDelete: (
    segment: TimelineSegment,
  ) => Promise<{ ok: boolean; detail: string | null }>;
  onReassign: () => void;
  onSplit: (
    segment: TimelineSegment,
    results: SplitResult[],
  ) => Promise<{ ok: boolean; detail: string | null; timeline?: Timeline }>;
  onSaved: (timeline: Timeline) => void;
  onDirtyChange: (dirty: boolean) => void;
}

export function SplitPanel({
  taskId,
  segment,
  originalSpeaker,
  speakerColorValue,
  timeline,
  names,
  editable,
  playing,
  playPos,
  onPlay,
  onDelete,
  onReassign,
  onSplit,
  onSaved,
  onDirtyChange,
}: SplitPanelProps) {
  const draft = useSplitDraft({
    segment,
    originalSpeaker,
    editable,
    onDirtyChange,
  });
  const {
    start,
    end,
    slots,
    targets,
    intervals,
    dirty,
    error,
    setError,
    discardDraft,
    addSlot,
  } = draft;
  const [saving, setSaving] = useState(false);

  const originalSpeakerInfo = timeline.speakers.find(
    (sp) => sp.label === originalSpeaker,
  );

  const playingThis =
    playing !== null &&
    playing.taskId === taskId &&
    playing.speaker === originalSpeaker &&
    playing.start === start &&
    playing.end === end;


  // 试听某个结果区间时高亮对应行：仅比较区间起止与原说话人。
  function isIntervalPlaying(iv: SplitInterval): boolean {
    return (
      playing !== null &&
      playing.taskId === taskId &&
      playing.speaker === originalSpeaker &&
      playing.start === iv.start &&
      playing.end === iv.end
    );
  }

  // 移动播放条：整段试听覆盖 [start, end]，区间试听只在对应区间内移动。
  const playBounds = splitPlayBounds(
    playing,
    taskId,
    originalSpeaker,
    start,
    end,
    intervals,
  );
  const playheadPos =
    playBounds !== null && playPos !== null
      ? Math.min(playBounds.to, Math.max(playBounds.from, playPos))
      : null;

  async function save() {
    const problem = validateSplitDraft(intervals, targets, originalSpeaker);
    if (problem) {
      setError(problem);
      return;
    }
    const results = toSplitResults(intervals, targets);
    setSaving(true);
    const outcome = await onSplit(segment, results);
    setSaving(false);
    if (!outcome.ok) {
      setError(outcome.detail ?? "保存切分失败");
      return;
    }
    if (outcome.timeline) {
      onSaved(outcome.timeline);
    }
    // 弹窗保持打开：清空当前草稿，停留的时段不残留间隔线/分配。
    discardDraft();
  }

  async function confirmDelete() {
    if (!editable || dirty || saving) return;
    if (
      !window.confirm(
        "删除该时段？该区间将不再归属任何说话人，不参与擦除与声纹校验。" +
          "\n\n删除后只能通过「放弃全部校正」恢复。",
      )
    ) {
      return;
    }
    setSaving(true);
    const outcome = await onDelete(segment);
    setSaving(false);
    if (!outcome.ok) {
      setError(outcome.detail ?? "删除时段失败");
      return;
    }
    // 删除成功后保持弹窗打开：选中时段由 DetailModal 依新时间轴自动切换。
  }

  return (
    <div className="operation-panel">
      <div className="operation-panel-head">
        <span className="dot" style={{ background: speakerColorValue }} />
        <strong>{originalSpeakerInfo ? shownName(names, originalSpeakerInfo) : originalSpeaker}</strong>
        <span className="operation-panel-meta">
          {fmtTime(start)} – {fmtTime(end)}
        </span>
        {playingThis && <span className="operation-panel-playing">试听中</span>}
      </div>
      <p className="operation-panel-hint">
        点击拉宽时间轴添加切分间隔线，拖动微调位置，悬停间隔线可删除；试听始终播放整个选中时段。
      </p>
      <div className="operation-panel-actions">
        <button type="button" onClick={() => onPlay({ start, end })}>
          试听整个选中时段
        </button>
        <button
          type="button"
          disabled={!editable}
          title={editable ? "把整个选中时段改派给另一个说话人" : "任务运行中或校正已锁定"}
          onClick={onReassign}
        >
          改派说话人
        </button>
        <button type="button" onClick={addSlot} disabled={!editable}>
          新建说话人槽位
        </button>
        <button
          type="button"
          className="danger"
          disabled={!editable || dirty || saving}
          title={
            !editable
              ? "任务运行中或校正已锁定"
              : dirty
                ? "有未保存的切分草稿，请先放弃草稿"
                : "删除该时段：区间不再归属任何说话人"
          }
          onClick={() => void confirmDelete()}
        >
          删除该时段
        </button>
        {dirty && (
          <button type="button" onClick={discardDraft} disabled={!editable}>
            放弃草稿
          </button>
        )}
      </div>
      <SplitTrack
        draft={draft}
        timeline={timeline}
        editable={editable}
        playBounds={playBounds}
        playheadPos={playheadPos}
      />
      <SplitIntervalList
        draft={draft}
        timeline={timeline}
        names={names}
        editable={editable}
        onPlay={onPlay}
        isIntervalPlaying={isIntervalPlaying}
      />
      {slots.length > 0 && (
        <p className="operation-panel-hint">
          新建说话人槽位保存后按现有规则顺延为 SPEAKER_XX；同一个槽位分配给多个区间时映射为同一个标签。
        </p>
      )}
      {error && <p className="error-text">{error}</p>}
      <div className="split-panel-actions">
        <button
          type="button"
          className="primary"
          onClick={() => void save()}
          disabled={saving || !editable}
        >
          {saving ? "保存中…" : "保存切分"}
        </button>
      </div>
    </div>
  );
}
