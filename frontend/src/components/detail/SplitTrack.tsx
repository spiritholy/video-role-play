// 切分草稿轨道：结果区间色块、可拖拽间隔线（悬停可删）与移动播放条。
import { fmtTime } from "../../lib/format";
import { isNewSlotTarget } from "../../lib/split";
import { speakerColor } from "../../lib/speakers";
import type { SplitDraft } from "../../hooks/useSplitDraft";
import type { Timeline } from "../../types";

export interface SplitTrackProps {
  draft: SplitDraft;
  timeline: Timeline;
  editable: boolean;
  playBounds: { from: number; to: number } | null;
  playheadPos: number | null;
}

export function SplitTrack({
  draft,
  timeline,
  editable,
  playBounds,
  playheadPos,
}: SplitTrackProps) {
  const {
    start,
    end,
    intervals,
    targets,
    markers,
    hovered,
    pct,
    durPct,
    markerLeft,
    setHovered,
    deleteMarker,
    onMarkerPointerDown,
    onMarkerPointerMove,
    onMarkerPointerUp,
    onTrackClick,
    trackRef,
    dragGuard,
    targetLabel,
  } = draft;
  return (
    <div className="split-track">
      <div
        className="split-track-canvas"
        ref={trackRef}
        onPointerDown={() => {
          dragGuard.current = false;
        }}
        onClick={onTrackClick}
      >
        {intervals.map((iv, i) => {
          const label = targets[i];
          const color =
            label && !isNewSlotTarget(label)
              ? speakerColor(timeline.speakers, label)
              : "#b8bec7";
          return (
            <div
              key={iv.key}
              className="split-interval"
              style={{
                left: pct(iv.start),
                width: durPct(iv.end - iv.start),
                background: color,
              }}
              title={`${label ? targetLabel(label) : "未指定"} · ${fmtTime(iv.start)} – ${fmtTime(iv.end)}`}
            />
          );
        })}
        {markers.map((m, index) => (
          <div
            key={`m${index}`}
            className="split-marker"
            style={{ left: markerLeft(m) }}
            onPointerDown={(e) => onMarkerPointerDown(e, index)}
            onPointerMove={(e) => onMarkerPointerMove(e, index)}
            onPointerUp={(e) => onMarkerPointerUp(e, index)}
            onMouseEnter={() => setHovered(index)}
            onMouseLeave={() => setHovered((h) => (h === index ? null : h))}
          >
            <div className="split-marker-line" />
            {hovered === index && editable && (
              <button
                type="button"
                className="split-marker-delete"
                title="删除此间隔线"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  deleteMarker(index);
                }}
              >
                ×
              </button>
            )}
          </div>
        ))}
        {playheadPos !== null && playBounds !== null && (
          <div className="split-playhead" style={{ left: pct(playheadPos) }}>
            <span className="split-playhead-time">{fmtTime(playheadPos)}</span>
          </div>
        )}
      </div>
      <div className="split-axis">
        <span>{fmtTime(start)}</span>
        <span>{fmtTime(end)}</span>
      </div>
    </div>
  );
}
