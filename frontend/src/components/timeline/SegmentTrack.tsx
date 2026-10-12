// 说话人时段轨道：主时间轴与详情弹窗总览共用，只是类名不同。
import type { MouseEvent as ReactMouseEvent } from "react";
import { fmtTime } from "../../lib/format";
import { MIN_SEG_PX, isSegmentActive } from "../../lib/segments";
import type { Playing, TimelineSegment, TimelineSpeaker } from "../../types";

export interface SegmentTrackProps {
  taskId: string;
  duration: number;
  speaker: TimelineSpeaker;
  color: string;
  playing: Playing | null;
  selectedKey: string | null;
  onSelect?: (speakerLabel: string, seg: TimelineSegment) => void;
  onBlankClick?: () => void;
  variant: "main" | "overview";
}

// 总览轨道中的单个说话人时段绘制：整片时长映射到可用轨道宽度，短时段保留
// 1px 最小可视宽度，超过容量时允许相邻时段视觉覆盖。主界面与详情弹窗顶部
// 总览共用同一套绘制，只是样式类名不同。
export function SegmentTrack({
  taskId,
  duration,
  speaker,
  color,
  playing,
  selectedKey,
  onSelect,
  onBlankClick,
  variant,
}: SegmentTrackProps) {
  const pct = (v: number) => `${((v / duration) * 100).toFixed(3)}%`;
  const classes =
    variant === "main"
      ? {
          track: "timeline-track",
          canvas: "timeline-track-canvas",
          wrap: "timeline-seg-wrap",
          seg: "timeline-seg",
        }
      : {
          track: "overview-track",
          canvas: "overview-track-canvas",
          wrap: "overview-seg-wrap",
          seg: "overview-seg",
        };
  return (
    <div className={classes.track} onClick={onBlankClick}>
      <div className={classes.canvas}>
        {speaker.segments.map((seg) => {
          const active = isSegmentActive(taskId, playing, speaker.label, seg);
          const selected = selectedKey === `${speaker.label}:${seg.index}`;
          return (
            <div
              key={seg.index}
              className={classes.wrap}
              style={{
                left: pct(seg.start),
                width: pct(seg.end - seg.start),
                minWidth: MIN_SEG_PX,
              }}
            >
              <button
                type="button"
                className={`${classes.seg}${active ? " playing" : ""}${selected ? " selected" : ""}`}
                style={{ background: color }}
                title={`${speaker.label} · ${fmtTime(seg.start)} – ${fmtTime(seg.end)}`}
                aria-label={`${speaker.label}，${fmtTime(seg.start)} 到 ${fmtTime(seg.end)}`}
                disabled={!onSelect}
                onClick={
                  onSelect
                    ? (event: ReactMouseEvent<HTMLButtonElement>) => {
                        event.stopPropagation();
                        onSelect(speaker.label, seg);
                      }
                    : undefined
                }
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
