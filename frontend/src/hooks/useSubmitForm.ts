// 发起处理任务的表单：先探测音轨，再带着选定的音轨与说话人数量提交。
import { useCallback, useState } from "react";
import { errorMessage } from "../api/client";
import { createTask, probeVideo } from "../api/tasks";
import type { ProbeResult } from "../types";

export interface SubmitFormState {
  videoPath: string;
  setVideoPath: (value: string) => void;
  probe: ProbeResult | null;
  selectedTrack: number;
  setSelectedTrack: (index: number) => void;
  numSpeakers: string;
  setNumSpeakers: (value: string) => void;
  formError: string | null;
  probing: boolean;
  handleProbe: () => Promise<void>;
  handleSubmit: (e: React.FormEvent) => Promise<void>;
}

export function useSubmitForm(deps: {
  refresh: () => Promise<void>;
}): SubmitFormState {
  const { refresh } = deps;
  const [videoPath, setVideoPath] = useState("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [selectedTrack, setSelectedTrack] = useState(0);
  const [numSpeakers, setNumSpeakers] = useState("");  // 空=让模型自动估计
  const [formError, setFormError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);

  const handleProbe = useCallback(async () => {
    setFormError(null);
    setProbe(null);
    if (!videoPath.trim()) {
      setFormError("请先填写源视频的本地路径");
      return;
    }
    setProbing(true);
    const result = await probeVideo(videoPath.trim());
    setProbing(false);
    if (!result.ok) {
      setFormError(errorMessage(result, "探测失败"));
      return;
    }
    const data = result.data;
    setProbe(data);
    setSelectedTrack(0);
    if (data.tracks.length === 0) setFormError("该视频没有音轨");
  }, [videoPath]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setFormError(null);
      if (!videoPath.trim()) {
        setFormError("请先填写源视频的本地路径");
        return;
      }
      const result = await createTask({
        video_path: videoPath.trim(),
        audio_track: probe && probe.tracks.length > 0 ? selectedTrack : null,
        num_speakers: numSpeakers.trim() ? Number(numSpeakers.trim()) : null,
      });
      if (!result.ok) {
        setFormError(errorMessage(result, "提交失败"));
        return;
      }
      refresh();
    },
    [numSpeakers, probe, refresh, selectedTrack, videoPath],
  );

  return {
    videoPath,
    setVideoPath,
    probe,
    selectedTrack,
    setSelectedTrack,
    numSpeakers,
    setNumSpeakers,
    formError,
    probing,
    handleProbe,
    handleSubmit,
  };
}
