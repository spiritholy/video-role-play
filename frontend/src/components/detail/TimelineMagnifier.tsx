// 放大镜：固定 8 秒窗口内所有说话人的时段条与中线指针。
import { fmtTime } from "../../lib/format";
import { magnifierSegmentGeometry } from "../../lib/magnifier";
import { speakerColor } from "../../lib/speakers";
import type { Timeline } from "../../types";

export interface TimelineMagnifierProps {
  timeline: Timeline;
  windowStart: number;
  windowEnd: number;
  windowSeconds: number;
  centerPct: number;
  hoverTime: number;
  left: number;
  top: number;
}

export function TimelineMagnifier({
  timeline,
  windowStart,
  windowEnd,
  windowSeconds,
  centerPct,
  hoverTime,
  left,
  top,
}: TimelineMagnifierProps) {
  const { inWindow, leftPct, widthPct } = magnifierSegmentGeometry(
    windowStart,
    windowEnd,
    windowSeconds,
  );
  return (
    <div
      className="magnifier"
      style={{ left, top }}
    >
      <div className="magnifier-label">
        <span>{fmtTime(windowStart)} – {fmtTime(windowEnd)}</span>
        <span className="magnifier-center-time">{fmtTime(hoverTime)}</span>
      </div>
      <div className="magnifier-canvas">
        {timeline.speakers.map((sp) => {
          const color = speakerColor(timeline.speakers, sp.label);
          return (
            <div key={sp.label} className="magnifier-lane">
              {sp.segments.filter(inWindow).map((seg) => (
                <div
                  key={seg.index}
                  className="magnifier-seg"
                  style={{
                    left: `${leftPct(seg).toFixed(3)}%`,
                    width: `${widthPct(seg).toFixed(3)}%`,
                    background: color,
                  }}
                />
              ))}
            </div>
          );
        })}
      </div>
      <div
        className="magnifier-center"
        style={{ left: `${centerPct.toFixed(3)}%` }}
      />
    </div>
  );
}
