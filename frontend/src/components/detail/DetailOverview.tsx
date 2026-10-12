// 详情弹窗顶部的全片总览：所有说话人时间对齐的轨道 + 悬停跟随的放大镜。
import { useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { fmtTime } from "../../lib/format";
import {
  magnifierPosition,
  magnifierTimeFromClientX,
  magnifierWindow,
} from "../../lib/magnifier";
import { shownName, speakerColor } from "../../lib/speakers";
import { SegmentTrack } from "../timeline/SegmentTrack";
import { TimelineMagnifier } from "./TimelineMagnifier";
import type { Playing, Timeline, TimelineSegment } from "../../types";

export interface DetailOverviewProps {
  taskId: string;
  timeline: Timeline;
  names: Record<string, string>;
  duration: number;
  playing: Playing | null;
  selectedKey: string | null;
  onSelect: (speakerLabel: string, seg: TimelineSegment) => void;
}

export function DetailOverview({
  taskId,
  timeline,
  names,
  duration,
  playing,
  selectedKey,
  onSelect,
}: DetailOverviewProps) {
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverClientX, setHoverClientX] = useState(0);
  const [hoverClientY, setHoverClientY] = useState(0);
  const overviewRef = useRef<HTMLDivElement | null>(null);

  // 固定 8 秒放大镜窗口：以悬停时间为中心，靠近片头/片尾时整体平移保持 8 秒。
  const { windowStart, windowEnd, windowSeconds, centerPct } = magnifierWindow(
    duration,
    hoverTime,
  );

  function handleOverviewMove(event: ReactMouseEvent<HTMLDivElement>) {
    if (duration <= 0) return;
    const el = overviewRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    setHoverClientX(event.clientX);
    setHoverClientY(event.clientY);
    // 时间映射以轨道列（时段实际绘制区域）宽度为基准，避免左侧名称列造成偏差。
    const trackEl = el.querySelector<HTMLElement>(".overview-track");
    setHoverTime(
      magnifierTimeFromClientX(
        event.clientX,
        rect,
        trackEl ? trackEl.getBoundingClientRect() : null,
        duration,
      ),
    );
  }

  const magnifier = magnifierPosition(
    hoverClientX,
    hoverClientY,
    centerPct,
    window.innerWidth,
  );

  return (
    <div
      className="detail-overview"
      ref={overviewRef}
      onMouseMove={handleOverviewMove}
      onMouseLeave={() => setHoverTime(null)}
    >
      <p className="detail-overview-title">
        全片总览（点击或 ← → 键选中时段；悬停显示固定 8 秒放大镜）
      </p>
      {timeline.speakers.map((sp) => {
        const color = speakerColor(timeline.speakers, sp.label);
        return (
          <div key={sp.label} className="overview-row">
            <span className="overview-chip">
              <span className="dot" style={{ background: color }} />
              {shownName(names, sp)}
              <span className="overview-chip-time">{fmtTime(sp.speech_time)}</span>
            </span>
            <SegmentTrack
              taskId={taskId}
              duration={duration}
              speaker={sp}
              color={color}
              playing={playing}
              selectedKey={selectedKey}
              onSelect={onSelect}
              variant="overview"
            />
          </div>
        );
      })}
      {duration > 0 && (
        <div className="overview-axis">
          <span />
          <div className="overview-axis-canvas">
            {Array.from({ length: 5 }, (_, i) => (
              <span key={i}>{fmtTime((duration * i) / 4)}</span>
            ))}
          </div>
        </div>
      )}
      {hoverTime !== null && windowSeconds > 0 && (
        <TimelineMagnifier
          timeline={timeline}
          windowStart={windowStart}
          windowEnd={windowEnd}
          windowSeconds={windowSeconds}
          centerPct={centerPct}
          hoverTime={hoverTime}
          left={magnifier.left}
          top={magnifier.top}
        />
      )}
    </div>
  );
}
