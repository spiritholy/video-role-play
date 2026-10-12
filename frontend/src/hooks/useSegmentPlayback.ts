// 说话人时段试听：整段切片播放，并驱动切分面板的移动播放条。
import { useCallback, useEffect, useRef, useState } from "react";
import { audioSegmentUrl } from "../api/urls";
import type { Playing, TimelineSegment } from "../types";

export interface SegmentPlayback {
  playing: Playing | null;
  // 试听播放进度（绝对秒）：切分面板用它绘制移动播放条。
  playPos: number | null;
  playSegment: (
    taskId: string,
    speaker: string,
    seg: Pick<TimelineSegment, "start" | "end">,
  ) => void;
}

export function useSegmentPlayback(): SegmentPlayback {
  const [playing, setPlaying] = useState<Playing | null>(null);
  const [playPos, setPlayPos] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      audioRef.current = null;
    };
  }, []);

  // 试听中的移动播放条位置：音频接口从 start 处切片，currentTime 是相对
  // 偏移，需加 playing.start 才是绝对时间；结束/出错时随 playing 复位。
  useEffect(() => {
    if (!playing) {
      setPlayPos(null);
      return;
    }
    let raf = 0;
    const tick = () => {
      const audio = audioRef.current;
      if (audio && !audio.paused && !audio.ended) {
        setPlayPos(Math.min(playing.end, playing.start + audio.currentTime));
        raf = requestAnimationFrame(tick);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const playSegment = useCallback(
    (
      taskId: string,
      speaker: string,
      seg: Pick<TimelineSegment, "start" | "end">,
    ) => {
      if (!audioRef.current) audioRef.current = new Audio();
      const audio = audioRef.current;
      audio.pause();
      audio.src = audioSegmentUrl(taskId, seg);
      audio.onended = () => setPlaying(null);
      audio.play().catch(() => setPlaying(null));
      setPlaying({ taskId, speaker, start: seg.start, end: seg.end });
    },
    [],
  );

  return { playing, playPos, playSegment };
}
