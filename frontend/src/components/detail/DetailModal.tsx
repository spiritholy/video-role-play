// 校正详情弹窗：顶部全片总览（含 8 秒放大镜），底部是选中时段的操作面板。
import { useEffect, useMemo, useState } from "react";
import { orderedSegmentsOf } from "../../lib/segments";
import { speakerColor } from "../../lib/speakers";
import { DetailOverview } from "./DetailOverview";
import { ReassignDialog } from "./ReassignDialog";
import { SplitPanel } from "./SplitPanel";
import type {
  Playing,
  SplitResult,
  Timeline,
  TimelineSegment,
} from "../../types";

export interface DetailModalProps {
  taskId: string;
  timeline: Timeline;
  names: Record<string, string>;
  playing: Playing | null;
  playPos: number | null;
  editable: boolean;
  onPlay: (
    taskId: string,
    speaker: string,
    seg: Pick<TimelineSegment, "start" | "end">,
  ) => void;
  onDelete: (
    segment: TimelineSegment,
  ) => Promise<{ ok: boolean; detail: string | null }>;
  onReassign: (segment: TimelineSegment, target: string) => Promise<boolean>;
  onSplit: (
    segment: TimelineSegment,
    results: SplitResult[],
  ) => Promise<{ ok: boolean; detail: string | null; timeline?: Timeline }>;
  initial: { speaker: string; segment: TimelineSegment };
  onClose: () => void;
}

export function DetailModal({
  taskId,
  timeline,
  names,
  playing,
  playPos,
  editable,
  onPlay,
  onDelete,
  onReassign,
  onSplit,
  initial,
  onClose,
}: DetailModalProps) {
  const [selected, setSelected] = useState<{
    speaker: string;
    segment: TimelineSegment;
  } | null>(initial);
  const [draftDirty, setDraftDirty] = useState(false);
  const [reassigning, setReassigning] = useState<TimelineSegment | null>(null);
  const [saving, setSaving] = useState(false);

  // 全片所有时段按时间排序，供 ← / → 键切换选中。
  const orderedSegments = useMemo(() => orderedSegmentsOf(timeline), [timeline]);

  // 选中时段是否仍存在于当前时间轴：按说话人 + 索引 + 起止时间四项比对，
  // 防止删除后剩余时段重排索引（旧索引被复用）时误判为仍选中。
  const selectedValid =
    selected !== null &&
    orderedSegments.some(
      (item) =>
        item.speaker === selected.speaker &&
        item.segment.index === selected.segment.index &&
        item.segment.start === selected.segment.start &&
        item.segment.end === selected.segment.end,
    );

  // 选中时段被删除后保持弹窗打开：自动切到紧随其后的时段，无则切到第一个；
  // 全部删完则进入空状态（selected = null）。
  useEffect(() => {
    if (selected === null || selectedValid) return;
    const after = orderedSegments.findIndex(
      (item) => item.segment.start > selected.segment.start,
    );
    const next = (after >= 0 ? orderedSegments[after] : orderedSegments[0]) ?? null;
    setDraftDirty(false);
    setSelected(next);
  }, [orderedSegments, selected, selectedValid]);

  const duration = timeline.duration;
  const selectedSpeaker =
    selected !== null
      ? timeline.speakers.find((sp) => sp.label === selected.speaker)
      : undefined;
  const selectedColor = selectedSpeaker
    ? speakerColor(timeline.speakers, selectedSpeaker.label)
    : "#7a8699";

  // 切分保存成功后保持弹窗打开：清空草稿标记，并把选中段更新为保存后
  // 时间轴中「与原段 start 相同的首段」（切分/整段改说话人后该 start 不变）。
  function handleSaved(updated: Timeline) {
    setDraftDirty(false);
    if (selected === null) return;
    const same = orderedSegmentsOf(updated).find(
      (item) => item.segment.start === selected.segment.start,
    );
    if (same) {
      setSelected({ speaker: same.speaker, segment: same.segment });
    }
  }

  // 未保存切分草稿时，关闭弹窗或切换时段都需要用户确认放弃。
  function confirmClose() {
    if (
      draftDirty &&
      !window.confirm("当前时段有未保存的切分草稿，放弃并退出校正详情？")
    ) {
      return;
    }
    onClose();
  }

  function handleSelect(speakerLabel: string, seg: TimelineSegment) {
    if (
      draftDirty &&
      !window.confirm("当前时段有未保存的切分草稿，放弃并切换到另一个时段？")
    ) {
      return;
    }
    setDraftDirty(false);
    setSelected({ speaker: speakerLabel, segment: seg });
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        confirmClose();
        return;
      }
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      if (selected === null) return;
      // 焦点在输入框/下拉等可编辑控件时，箭头键留给控件本身。
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT")
      ) {
        return;
      }
      event.preventDefault();
      const currentIndex = orderedSegments.findIndex(
        (item) =>
          item.speaker === selected.speaker &&
          item.segment.index === selected.segment.index,
      );
      const next = orderedSegments[currentIndex + (event.key === "ArrowLeft" ? -1 : 1)];
      if (next) handleSelect(next.speaker, next.segment);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [confirmClose, handleSelect, orderedSegments, selected]);

  return (
    <div
      className="detail-modal-backdrop"
      role="presentation"
      onMouseDown={confirmClose}
    >
      <div
        className="detail-modal"
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="detail-modal-head">
          <h2>说话人时段校正</h2>
          <button
            type="button"
            className="detail-modal-close"
            aria-label="关闭"
            onClick={confirmClose}
          >
            ×
          </button>
        </div>
        <DetailOverview
          taskId={taskId}
          timeline={timeline}
          names={names}
          duration={duration}
          playing={playing}
          selectedKey={
            selected ? `${selected.speaker}:${selected.segment.index}` : null
          }
          onSelect={handleSelect}
        />
        {selected === null ? (
          <div className="operation-panel">
            <p className="operation-panel-hint">
              已删除全部时段，没有可编辑的时段。
            </p>
          </div>
        ) : (
          <SplitPanel
            key={`${selected.speaker}:${selected.segment.index}`}
            taskId={taskId}
            segment={selected.segment}
            originalSpeaker={selected.speaker}
            speakerColorValue={selectedColor}
            timeline={timeline}
            names={names}
            editable={editable}
            playing={playing}
            playPos={playPos}
            onPlay={(seg) => onPlay(taskId, selected.speaker, seg)}
            onDelete={onDelete}
            onReassign={() => setReassigning(selected.segment)}
            onSplit={onSplit}
            onSaved={handleSaved}
            onDirtyChange={setDraftDirty}
          />
        )}
        {reassigning && (
          <ReassignDialog
            segment={reassigning}
            timeline={timeline}
            names={names}
            saving={saving}
            onCancel={() => setReassigning(null)}
            onSave={async (target) => {
              setSaving(true);
              const ok = await onReassign(reassigning, target);
              setSaving(false);
              if (ok) setReassigning(null);
            }}
          />
        )}
      </div>
    </div>
  );
}
