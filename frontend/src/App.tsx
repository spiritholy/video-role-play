import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { audioUrl } from "./api/urls";
import {
  dirName,
  fileName,
  fmtTime,
  stageSummary,
  trackLabel,
} from "./lib/format";
import { STEM_LABELS } from "./lib/labels";
import {
  MIN_SEG_PX,
  firstSegment,
  isSegmentActive,
  orderedSegmentsOf,
} from "./lib/segments";
import { speakerColor, shownName } from "./lib/speakers";
import {
  MIN_SPLIT_SECONDS,
  NEW_SLOT_TOKEN,
  SPLIT_EPS,
  isNewSlotTarget,
  splitIntervals,
} from "./lib/split";
import { workspaceKey } from "./lib/workspace";
import type {
  Playing,
  ProbeResult,
  SplitInterval,
  SplitResult,
  Task,
  Timeline,
  TimelineOrError,
  TimelineSegment,
  TimelineSpeaker,
  TimelineUpdate,
  TranscriptSample,
  Transcripts,
  TranscriptsOrError,
  VoiceprintCheckState,
  VoiceprintList,
  VoiceprintSuggestion,
  WorkspaceItem,
  WorkspaceList,
} from "./types";

export default function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [workspaces, setWorkspaces] = useState<WorkspaceList | null>(null);
  const [voiceprints, setVoiceprints] = useState<VoiceprintList | null>(null);
  const [voiceprintError, setVoiceprintError] = useState<string | null>(null);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [timelines, setTimelines] = useState<Record<string, TimelineOrError>>({});
  // 说话人命名（工作区元数据的本地镜像）：工作区 → 编号标签 → 名字。
  // 按工作区键控：命名是工作区级数据，同源多任务卡片共享同一份显示。
  const [speakerNames, setSpeakerNames] = useState<
    Record<string, Record<string, string>>
  >({});
  const [transcripts, setTranscripts] = useState<Record<string, TranscriptsOrError>>({});
  // 命名入库失败警告（任务 → 文案）：仅前端会话态，下次命名成功或刷新即清除。
  const [ingestWarnings, setIngestWarnings] = useState<Record<string, string | null>>({});
  const [voiceprintBusy, setVoiceprintBusy] = useState<Record<string, boolean>>({});
  const [checkStates, setCheckStates] = useState<Record<string, VoiceprintCheckState>>({});
  const [checkError, setCheckError] = useState<string | null>(null);
  const [decisionBusy, setDecisionBusy] = useState<Record<string, boolean>>({});
  const [playing, setPlaying] = useState<Playing | null>(null);
  // 试听播放进度（绝对秒）：切分面板用它绘制移动播放条。
  const [playPos, setPlayPos] = useState<number | null>(null);
  const [correctionError, setCorrectionError] = useState<Record<string, string | null>>({});
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const requestedTranscripts = useRef<Set<string>>(new Set());
  const requestedChecks = useRef<Set<string>>(new Set());
  const [videoPath, setVideoPath] = useState("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [selectedTrack, setSelectedTrack] = useState(0);
  const [numSpeakers, setNumSpeakers] = useState("");  // 空=让模型自动估计
  const [formError, setFormError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);

  const maybeFetchTimeline = useCallback(
    async (t: Task) => {
      if (!t.stages.some((s) => s.id === "diarize" && s.status === "done")) return;
      try {
        const resp = await fetch(`/api/tasks/${t.id}/diarization`);
        if (!resp.ok) {
          setTimelines((prev) => ({ ...prev, [t.id]: "error" }));
          return;
        }
        const data: Timeline = await resp.json();
        setTimelines((prev) => ({ ...prev, [t.id]: data }));
        const wsKey = workspaceKey(t);
        // 本地镜像只收手工命名：自动认出来的名字带 auto 标记，由时间轴接口
        // 的 name/auto 直接展示，不冒充手工命名（否则「自动」徽标会丢失）
        setSpeakerNames((prev) => ({
          ...prev,
          [wsKey]: Object.fromEntries(
            data.speakers
              .filter((s) => s.name && !s.auto)
              .map((s) => [s.label, s.name as string]),
          ),
        }));
      } catch {
        setTimelines((prev) => ({ ...prev, [t.id]: "error" }));
      }
    },
    [],
  );

  const applyTimeline = useCallback((data: TimelineUpdate) => {
    setTimelines((prev) => ({ ...prev, [data.taskId]: data.timeline }));
    setSpeakerNames((prev) => ({
      ...prev,
      [data.workspaceKey]: Object.fromEntries(
        data.timeline.speakers
          .filter((s) => s.name && !s.auto)
          .map((s) => [s.label, s.name as string]),
      ),
    }));
  }, []);

  const reassignSegment = useCallback(
    async (task: Task, segment: TimelineSegment, target: string) => {
      setCorrectionError((prev) => ({ ...prev, [task.id]: null }));
      try {
        const resp = await fetch(
          `/api/tasks/${task.id}/segments/${segment.index}/speaker`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ target }),
          },
        );
        const payload = await resp.json().catch(() => null);
        if (!resp.ok) {
          setCorrectionError((prev) => ({
            ...prev,
            [task.id]: payload?.detail ?? "改派说话人失败",
          }));
          return false;
        }
        applyTimeline({
          taskId: task.id,
          workspaceKey: workspaceKey(task),
          timeline: payload,
        });
        return true;
      } catch {
        setCorrectionError((prev) => ({ ...prev, [task.id]: "无法连接后端服务" }));
        return false;
      }
    },
    [applyTimeline],
  );

  const splitSegment = useCallback(
    async (
      task: Task,
      segment: TimelineSegment,
      results: SplitResult[],
    ): Promise<{ ok: boolean; detail: string | null; timeline?: Timeline }> => {
      try {
        const resp = await fetch(
          `/api/tasks/${task.id}/segments/${segment.index}/split`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ results }),
          },
        );
        const payload = await resp.json().catch(() => null);
        if (!resp.ok) {
          return { ok: false, detail: payload?.detail ?? "保存切分失败" };
        }
        applyTimeline({
          taskId: task.id,
          workspaceKey: workspaceKey(task),
          timeline: payload,
        });
        return { ok: true, detail: null, timeline: payload };
      } catch {
        return { ok: false, detail: "无法连接后端服务" };
      }
    },
    [applyTimeline],
  );

  const deleteSegment = useCallback(
    async (
      task: Task,
      segment: TimelineSegment,
    ): Promise<{ ok: boolean; detail: string | null }> => {
      setCorrectionError((prev) => ({ ...prev, [task.id]: null }));
      try {
        const resp = await fetch(
          `/api/tasks/${task.id}/segments/${segment.index}`,
          { method: "DELETE" },
        );
        const payload = await resp.json().catch(() => null);
        if (!resp.ok) {
          const detail = payload?.detail ?? "删除时段失败";
          setCorrectionError((prev) => ({ ...prev, [task.id]: detail }));
          return { ok: false, detail };
        }
        applyTimeline({
          taskId: task.id,
          workspaceKey: workspaceKey(task),
          timeline: payload,
        });
        return { ok: true, detail: null };
      } catch {
        setCorrectionError((prev) => ({ ...prev, [task.id]: "无法连接后端服务" }));
        return { ok: false, detail: "无法连接后端服务" };
      }
    },
    [applyTimeline],
  );

  const discardCorrections = useCallback(
    async (task: Task) => {
      setCorrectionError((prev) => ({ ...prev, [task.id]: null }));
      try {
        const resp = await fetch(`/api/tasks/${task.id}/corrections`, {
          method: "DELETE",
        });
        const payload = await resp.json().catch(() => null);
        if (!resp.ok) {
          setCorrectionError((prev) => ({
            ...prev,
            [task.id]: payload?.detail ?? "放弃校正失败",
          }));
          return;
        }
        applyTimeline({
          taskId: task.id,
          workspaceKey: workspaceKey(task),
          timeline: payload,
        });
      } catch {
        setCorrectionError((prev) => ({ ...prev, [task.id]: "无法连接后端服务" }));
      }
    },
    [applyTimeline],
  );

  const renameSpeaker = useCallback(
    async (
      task: Task,
      label: string,
      rawName: string,
    ): Promise<string | null | undefined> => {
      const wsKey = workspaceKey(task);
      try {
        const resp = await fetch(
          `/api/tasks/${task.id}/speakers/${encodeURIComponent(label)}/name`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: rawName }),
          },
        );
        if (!resp.ok) return undefined;
        const data: { label: string; name: string | null } = await resp.json();
        setIngestWarnings((prev) => ({ ...prev, [task.id]: null }));
        setSpeakerNames((prev) => {
          const forWs = { ...(prev[wsKey] ?? {}) };
          if (data.name) forWs[label] = data.name;
          else delete forWs[label];
          return { ...prev, [wsKey]: forWs };
        });
        return data.name;
      } catch {
        return undefined;
      }
    },
    [],
  );

  const maybeFetchTranscripts = useCallback(
    async (t: Task) => {
      if (!t.stages.some((s) => s.id === "transcribe" && s.status === "done")) return;
      if (requestedTranscripts.current.has(t.id)) return;
      requestedTranscripts.current.add(t.id);
      try {
        const resp = await fetch(`/api/tasks/${t.id}/transcripts`);
        if (!resp.ok) {
          setTranscripts((prev) => ({ ...prev, [t.id]: "error" }));
          return;
        }
        const data: Transcripts = await resp.json();
        setTranscripts((prev) => ({ ...prev, [t.id]: data }));
      } catch {
        setTranscripts((prev) => ({ ...prev, [t.id]: "error" }));
      }
    },
    [],
  );

  const maybeFetchCheck = useCallback(async (t: Task) => {
    if (!t.stages.some((s) => s.id === "diarize" && s.status === "done")) return;
    const active = t.voiceprint_check && ["queued", "running"].includes(t.voiceprint_check.state);
    if (!active && requestedChecks.current.has(t.id)) return;
    requestedChecks.current.add(t.id);
    try {
      const resp = await fetch(`/api/tasks/${t.id}/voiceprint-check`);
      if (!resp.ok) return;
      const data: VoiceprintCheckState = await resp.json();
      setCheckStates((prev) => ({ ...prev, [t.id]: data }));
    } catch {
      // 下一轮轮询重试
    }
  }, []);

  const refreshWorkspaces = useCallback(async () => {
    try {
      const resp = await fetch("/api/workspaces");
      if (!resp.ok) return;
      setWorkspaces(await resp.json());
    } catch {
      // 后端暂不可达时保留旧列表，下一轮重试
    }
  }, []);

  const refreshVoiceprints = useCallback(async () => {
    try {
      const resp = await fetch("/api/voiceprints");
      if (!resp.ok) return;
      setVoiceprints(await resp.json());
    } catch {
      // 后端暂不可达时保留旧列表，下一轮重试
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const resp = await fetch("/api/tasks");
      const data = await resp.json();
      setTasks(data.tasks);
      data.tasks.forEach(maybeFetchTimeline);
      data.tasks.forEach(maybeFetchTranscripts);
      data.tasks.forEach(maybeFetchCheck);
    } catch {
      // 后端暂不可达时保留旧列表，下一轮重试
    }
    refreshWorkspaces();
    refreshVoiceprints();
  }, [maybeFetchTimeline, maybeFetchTranscripts, maybeFetchCheck, refreshWorkspaces, refreshVoiceprints]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [refresh]);

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

  function playSegment(
    taskId: string,
    speaker: string,
    seg: Pick<TimelineSegment, "start" | "end">,
  ) {
    if (!audioRef.current) audioRef.current = new Audio();
    const audio = audioRef.current;
    audio.pause();
    audio.src = `/api/tasks/${taskId}/audio-segment?start=${seg.start}&end=${seg.end}`;
    audio.onended = () => setPlaying(null);
    audio.play().catch(() => setPlaying(null));
    setPlaying({ taskId, speaker, start: seg.start, end: seg.end });
  }

  async function handleProbe() {
    setFormError(null);
    setProbe(null);
    if (!videoPath.trim()) {
      setFormError("请先填写源视频的本地路径");
      return;
    }
    setProbing(true);
    try {
      const resp = await fetch("/api/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: videoPath.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setFormError(data.detail ?? "探测失败");
        return;
      }
      setProbe(data);
      setSelectedTrack(0);
      if (data.tracks.length === 0) setFormError("该视频没有音轨");
    } catch {
      setFormError("无法连接后端服务");
    } finally {
      setProbing(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);
    if (!videoPath.trim()) {
      setFormError("请先填写源视频的本地路径");
      return;
    }
    try {
      const resp = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: videoPath.trim(),
          audio_track: probe && probe.tracks.length > 0 ? selectedTrack : null,
          num_speakers: numSpeakers.trim() ? Number(numSpeakers.trim()) : null,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setFormError(data.detail ?? "提交失败");
        return;
      }
      refresh();
    } catch {
      setFormError("无法连接后端服务");
    }
  }

  async function handleCancel(id: string) {
    await fetch(`/api/tasks/${id}/cancel`, { method: "POST" });
    refresh();
  }

  async function handleResume(id: string) {
    setResumeError(null);
    try {
      const resp = await fetch(`/api/tasks/${id}/resume`, { method: "POST" });
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        setResumeError(data?.detail ?? "续跑失败");
        return;
      }
    } catch {
      setResumeError("无法连接后端服务");
      return;
    }
    refresh();
  }

  async function handleResumeWorkspace(w: WorkspaceItem) {
    setWorkspaceError(null);
    try {
      const resp = await fetch(
        `/api/workspaces/${encodeURIComponent(w.name)}/resume`,
        { method: "POST" },
      );
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        setWorkspaceError(data?.detail ?? "续跑失败");
        return;
      }
    } catch {
      setWorkspaceError("无法连接后端服务");
      return;
    }
    refresh();
  }

  async function handleOpenWorkspace(w: WorkspaceItem) {
    setWorkspaceError(null);
    try {
      const resp = await fetch(
        `/api/workspaces/${encodeURIComponent(w.name)}/open`,
        { method: "POST" },
      );
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        setWorkspaceError(data?.detail ?? "打开擦除界面失败");
        return;
      }
    } catch {
      setWorkspaceError("无法连接后端服务");
      return;
    }
    refresh();
  }

  async function handleDeleteWorkspace(w: WorkspaceItem) {
    const confirmed = window.confirm(
      `删除工作区「${w.name}」？\n\n释放 ${w.size_label}，` +
        `全部中间产物一并删除。\n再次处理该源视频需要从头分析（数小时）。`,
    );
    if (!confirmed) return;
    setWorkspaceError(null);
    try {
      const resp = await fetch(`/api/workspaces/${encodeURIComponent(w.name)}`, {
        method: "DELETE",
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        setWorkspaceError(data?.detail ?? "删除工作区失败");
        return;
      }
    } catch {
      setWorkspaceError("无法连接后端服务");
      return;
    }
    refresh();
  }

  async function handleStoreVoiceprint(task: Task, label: string) {
    const key = `${task.id}:${label}`;
    if (voiceprintBusy[key]) return;
    const timeline = timelines[task.id];
    const speaker =
      timeline === "error" || timeline === undefined
        ? undefined
        : timeline.speakers.find((sp) => sp.label === label);
    const name =
      speakerNames[workspaceKey(task)]?.[label] ?? speaker?.name ?? null;
    if (!name) {
      setIngestWarnings((prev) => ({
        ...prev,
        [task.id]: "请先命名说话人，再存入声纹库",
      }));
      return;
    }

    setVoiceprintBusy((prev) => ({ ...prev, [key]: true }));
    setIngestWarnings((prev) => ({ ...prev, [task.id]: null }));
    try {
      const resp = await fetch(
        `/api/tasks/${task.id}/speakers/${encodeURIComponent(label)}/voiceprint`,
        { method: "POST" },
      );
      const data = await resp.json().catch(() => null);
      if (!resp.ok) {
        setIngestWarnings((prev) => ({
          ...prev,
          [task.id]: data?.detail ?? "存入声纹库失败",
        }));
        return;
      }
      setIngestWarnings((prev) => ({
        ...prev,
        [task.id]: data?.voiceprint_error
          ? `声纹入库失败：${data.voiceprint_error}`
          : null,
      }));
    } catch {
      setIngestWarnings((prev) => ({
        ...prev,
        [task.id]: "无法连接后端服务",
      }));
    } finally {
      setVoiceprintBusy((prev) => ({ ...prev, [key]: false }));
      refreshVoiceprints();
    }
  }

  async function handleDeleteVoiceprint(name: string) {
    const confirmed = window.confirm(
      `从声纹库删除「${name}」？\n\n后续剧集将不再自动识别该人物。`,
    );
    if (!confirmed) return;
    setVoiceprintError(null);
    try {
      const resp = await fetch(
        `/api/voiceprints/${encodeURIComponent(name)}`,
        { method: "DELETE" },
      );
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        setVoiceprintError(data?.detail ?? "删除声纹失败");
        return;
      }
    } catch {
      setVoiceprintError("无法连接后端服务");
      return;
    }
    refreshVoiceprints();
  }

  function renderTimeline(t: Task) {
    const timeline = timelines[t.id];
    if (timeline === undefined) {
      return <p className="timeline-hint">正在加载说话人时间轴…</p>;
    }
    if (timeline === "error") {
      return <p className="timeline-hint">说话人时间轴加载失败</p>;
    }

    const transcribeStage = t.stages.find((s) => s.id === "transcribe");
    const data = transcripts[t.id];
    let samplesHint: string | null = null;
    let samplesBySpeaker: Record<string, TranscriptSample[]> = {};
    if (data === undefined) {
      if (
        transcribeStage &&
        (transcribeStage.status === "pending" ||
          transcribeStage.status === "running")
      ) {
        samplesHint = "样本转写中…";
      } else if (transcribeStage?.status === "done") {
        samplesHint = "正在加载样本台词…";
      }
    } else if (data === "error") {
      samplesHint = "样本台词加载失败";
    } else {
      for (const s of data.samples) {
        (samplesBySpeaker[s.speaker] ??= []).push(s);
      }
    }

    const ingestWarning = ingestWarnings[t.id];
    return (
      <>
        <SpeakerTimeline
          taskId={t.id}
          timeline={timeline}
          names={speakerNames[workspaceKey(t)] ?? {}}
          samplesBySpeaker={samplesBySpeaker}
          samplesHint={samplesHint}
          playing={playing}
          playPos={playPos}
          onPlay={playSegment}
          onDelete={(segment) => deleteSegment(t, segment)}
          onRename={(label, name) => renameSpeaker(t, label, name)}
          onStoreVoiceprint={(label) => handleStoreVoiceprint(t, label)}
          voiceprintBusy={(rowLabel) => Boolean(voiceprintBusy[`${t.id}:${rowLabel}`])}
          onReassign={(segment, target) => reassignSegment(t, segment, target)}
          onSplit={(segment, results) => splitSegment(t, segment, results)}
          onDiscard={() => discardCorrections(t)}
          correctionError={correctionError[t.id] ?? null}
        />
        {ingestWarning && <p className="error-text">{ingestWarning}</p>}
        {t.state === "completed" && (
          <EraseForm
            task={t}
            speakers={timeline.speakers}
            names={speakerNames[workspaceKey(t)] ?? {}}
          />
        )}
      </>
    );
  }

  async function startVoiceprintCheck(t: Task) {
    setCheckError(null);
    try {
      const resp = await fetch(`/api/tasks/${t.id}/voiceprint-check`, { method: "POST" });
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        setCheckError(data?.detail ?? "声纹校验启动失败");
        return;
      }
      const data = await resp.json();
      if (data.voiceprint_check) {
        setCheckStates((prev) => ({
          ...prev,
          [t.id]: { ...data.voiceprint_check, result: null },
        }));
      }
    } catch {
      setCheckError("无法连接后端服务");
    }
  }

  async function cancelVoiceprintCheck(t: Task) {
    setCheckError(null);
    try {
      const resp = await fetch(`/api/tasks/${t.id}/voiceprint-check`, { method: "DELETE" });
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        setCheckError(data?.detail ?? "取消声纹校验失败");
      }
    } catch {
      setCheckError("无法连接后端服务");
    }
  }

  async function decideSuggestion(t: Task, index: number, decision: "accept" | "ignore") {
    const key = `${t.id}:${index}`;
    if (decisionBusy[key]) return;
    setDecisionBusy((prev) => ({ ...prev, [key]: true }));
    setCheckError(null);
    try {
      const resp = await fetch(
        `/api/tasks/${t.id}/voiceprint-check/suggestions/${index}/decision`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ decision }),
        },
      );
      const data = await resp.json().catch(() => null);
      if (!resp.ok) {
        setCheckError(data?.detail ?? "建议裁决失败");
        return;
      }
      setCheckStates((prev) => ({
        ...prev,
        [t.id]: {
          ...(prev[t.id] ?? { state: "completed", progress: 1, error: null }),
          state: "completed",
          progress: 1,
          error: null,
          result: data.result,
        },
      }));
      applyTimeline({ taskId: t.id, workspaceKey: workspaceKey(t), timeline: data.timeline });
    } catch {
      setCheckError("无法连接后端服务");
    } finally {
      setDecisionBusy((prev) => ({ ...prev, [key]: false }));
    }
  }

  function suggestionText(s: VoiceprintSuggestion): string {
    const pct = (v?: number) => (v == null ? "" : ` · 相似度 ${Math.round(v * 100)}%`);
    if (s.type === "rename") return `建议把 ${s.label} 命名为「${s.name}」${pct(s.similarity)}`;
    if (s.type === "merge") return `建议合并同名簇 ${s.labels?.join("、")} 为 ${s.canonical}`;
    return `建议把 ${s.start?.toFixed(1)}s–${s.end?.toFixed(1)}s 从 ${s.from_label} 改派给「${s.name}」${pct(s.similarity)}`;
  }

  function renderVoiceprintCheck(t: Task) {
    const check = t.voiceprint_check && ["queued", "running"].includes(t.voiceprint_check.state)
      ? t.voiceprint_check
      : checkStates[t.id];
    if (!check) return null;
    const active = check.state === "queued" || check.state === "running";
    return (
      <div className="voiceprint-check">
        <div className="voiceprint-check-head">
          <strong>声纹库校验</strong>
          {active ? (
            <>
              <span>{Math.round((check.progress ?? 0) * 100)}%</span>
              <button className="cancel" onClick={() => cancelVoiceprintCheck(t)}>取消</button>
            </>
          ) : (
            <button className="resume" onClick={() => startVoiceprintCheck(t)}>用声纹库校验</button>
          )}
        </div>
        {active && (
          <div className="stage-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((check.progress ?? 0) * 100)}>
            <span className="stage-progress-fill" style={{ width: `${Math.round((check.progress ?? 0) * 100)}%` }} />
          </div>
        )}
        {check.state === "failed" && <p className="error-text">{check.error ?? "声纹校验失败"}</p>}
        {check.state === "cancelled" && <p className="timeline-hint">声纹校验已取消。</p>}
        {check.state === "not_run" && <p className="timeline-hint">尚未校验。此操作只生成建议，不会直接修改时段。</p>}
        {check.result?.status === "invalidated" && (
          <p className="correction-warning">声纹校验建议已作废：说话人分离已重新进行。</p>
        )}
        {check.result?.suggestions?.length ? (
          <ul className="suggestion-list">
            {check.result.suggestions.map((s, i) => (
              <li key={i} className="suggestion">
                <span className="suggestion-type">
                  {s.type === "rename" ? "命名" : s.type === "merge" ? "合并" : "改派"}
                </span>
                <span>{suggestionText(s)}</span>
                <span className={`suggestion-state ${s.decision ?? "pending"}`}>
                  {s.decision === "accepted" ? "已接受" : s.decision === "ignored" ? "已忽略" : "待确认"}
                </span>
                {check.result?.status === "invalidated" ? null : (
                  <span className="suggestion-actions">
                    <button
                      type="button"
                      className="resume"
                      disabled={Boolean(decisionBusy[`${t.id}:${i}`])}
                      onClick={() => decideSuggestion(t, i, "accept")}
                    >
                      接受
                    </button>
                    <button
                      type="button"
                      className="cancel"
                      disabled={Boolean(decisionBusy[`${t.id}:${i}`])}
                      onClick={() => decideSuggestion(t, i, "ignore")}
                    >
                      忽略
                    </button>
                  </span>
                )}
              </li>
            ))}
          </ul>
        ) : check.state === "completed" ? (
          <p className="timeline-hint">校验完成，未产生建议。</p>
        ) : null}
      </div>
    );
  }

  return (
    <main className="app">
      <h1>视频人声擦除</h1>

      <form className="submit-form" onSubmit={handleSubmit}>
        <label htmlFor="video-path">源视频路径（服务器本地路径）</label>
        <div className="path-row">
          <input
            id="video-path"
            type="text"
            placeholder="/home/user/videos/电影.mkv"
            value={videoPath}
            onChange={(e) => setVideoPath(e.target.value)}
          />
          <button type="button" onClick={handleProbe} disabled={probing}>
            {probing ? "探测中…" : "探测音轨"}
          </button>
        </div>

        {probe && probe.tracks.length > 0 && (
          <div className="track-row">
            <label htmlFor="audio-track">
              音轨（共 {probe.tracks.length} 条，默认第一条）
            </label>
            <select
              id="audio-track"
              value={selectedTrack}
              onChange={(e) => setSelectedTrack(Number(e.target.value))}
            >
              {probe.tracks.map((t) => (
                <option key={t.index} value={t.index}>
                  {trackLabel(t)}
                </option>
              ))}
            </select>
          </div>
        )}

        <label htmlFor="num-speakers">
          说话人数量（可选，留空=自动估计；原片人数与自动结果不符时在此指定）
        </label>
        <input
          id="num-speakers"
          type="number"
          min={1}
          max={20}
          placeholder="如 4"
          value={numSpeakers}
          onChange={(e) => setNumSpeakers(e.target.value)}
        />

        {formError && <p className="error-text">{formError}</p>}

        <button type="submit" className="primary">
          提交处理任务
        </button>
      </form>

      <section className="tasks">
        <h2>处理任务</h2>
        {resumeError && <p className="error-text">{resumeError}</p>}
        {checkError && <p className="error-text">{checkError}</p>}
        {tasks.length === 0 && <p className="empty">还没有任务，提交一个源视频开始。</p>}
        {tasks.map((t) => {
          const cancellable = t.state === "queued" || t.state === "running";
          const resumable = t.state === "failed" || t.state === "cancelled";
          return (
            <div key={t.id} className={`task-card state-${t.state}`}>
              <div className="task-head">
                <span className="task-name">{fileName(t.video_path)}</span>
                <span className={`badge badge-${t.state}`}>{t.state_label}</span>
                {cancellable && (
                  <button className="cancel" onClick={() => handleCancel(t.id)}>
                    取消
                  </button>
                )}
                {resumable && (
                  <button className="resume" onClick={() => handleResume(t.id)}>
                    继续
                  </button>
                )}
              </div>
              <div className="task-meta">
                <span>{t.video_path}</span>
                {t.audio_track != null && <span>音轨 #{t.audio_track + 1}</span>}
              </div>
              <div className="task-stages">
                <span>{stageSummary(t)}</span>
                <span className="stage-chips">
                  {t.stages.map((s) => (
                    <span key={s.id} className={`chip chip-${s.status}`}>
                      {s.name}
                      {s.status === "running" && (
                        <span
                          className="stage-progress"
                          role="progressbar"
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={s.progress == null ? undefined : Math.round(s.progress * 100)}
                        >
                          <span
                            className="stage-progress-fill"
                            style={{ width: `${Math.round((s.progress ?? 0) * 100)}%` }}
                          />
                        </span>
                      )}
                    </span>
                  ))}
                </span>
              </div>
              {resumable && (
                <p className="task-hint">
                  已完成阶段的产物保留在工作区，点“继续”从断点阶段跑起，不重算分离与说话人分离。
                </p>
              )}
              {t.stages.filter((s) => s.status === "done" && Object.keys(s.outputs).length > 0)
                .map((s) => (
                  <div key={s.id} className="stems">
                    {Object.entries(s.outputs).map(([role, artifact]) => (
                      <div key={role} className="stem">
                        <span className="stem-label">
                          {STEM_LABELS[role] ?? role}
                        </span>
                        <audio
                          controls
                          preload="none"
                          src={audioUrl(t.id, artifact)}
                        />
                      </div>
                    ))}
                  </div>
                ))}
              {t.stages.some((s) => s.id === "diarize" && s.status === "done") &&
                renderVoiceprintCheck(t)}
              {t.stages.some((s) => s.id === "diarize" && s.status === "done") &&
                renderTimeline(t)}
              {t.error && <p className="error-text">失败原因：{t.error}</p>}
              {t.output_path && (
                <p className="output-text">成品：{t.output_path}</p>
              )}
            </div>
          );
        })}
      </section>

      <section className="workspaces">
        <h2>工作区{workspaces ? `（共占用 ${workspaces.total_label}）` : ""}</h2>
        {workspaceError && <p className="error-text">{workspaceError}</p>}

        <div className="voiceprint-panel">
          <div className="voiceprint-title">
            人物声纹库
            {voiceprints && voiceprints.names.length > 0 && (
              <span className="voiceprint-meta">
                已积累 {voiceprints.names.length} 个人物 · 库版本 {voiceprints.revision}
              </span>
            )}
          </div>
          <p className="voiceprint-hint">
            在说话人列表中显式存入声纹；跨视频复用需点击「用声纹库校验」。
            校验只给出命名、合并、改派建议，经你确认后才会修改。误入库的人物可在此删除。
          </p>
          {voiceprintError && <p className="error-text">{voiceprintError}</p>}
          {voiceprints && voiceprints.names.length === 0 && (
            <p className="empty">声纹库为空。在说话人列表中显式存入声纹。</p>
          )}
          {voiceprints?.names.map((n) => (
            <div key={n.name} className="voiceprint-row">
              <span className="voiceprint-name">{n.name}</span>
              <span className="voiceprint-count">{n.vectors_count} 个声纹</span>
              {n.source_workspace && (
                <span className="voiceprint-source">来自 {n.source_workspace}</span>
              )}
              <button
                className="delete"
                title={`从声纹库删除「${n.name}」`}
                onClick={() => handleDeleteVoiceprint(n.name)}
              >
                删除
              </button>
            </div>
          ))}
        </div>
        {workspaces && workspaces.workspaces.length === 0 && (
          <p className="empty">
            还没有工作区。任务开始运行后，中间产物会写入工作区目录，可在此手动删除释放磁盘。
          </p>
        )}
        {workspaces?.workspaces.map((w) => {
          const hasCompletedTask = tasks.some(
            (t) => t.workspace_dir === w.path && t.state === "completed",
          );
          return (
          <div key={w.name} className="workspace-row">
            <div className="workspace-main">
              <span className="workspace-name">{w.name}</span>
              <span className="workspace-size">{w.size_label}</span>
              <span className={`workspace-tag${w.reusable ? " tag-reusable" : ""}`}>
                {w.reusable ? "分析可复用" : "分析未完成"}
              </span>
              {w.busy && <span className="workspace-tag tag-busy">任务使用中</span>}
              {w.resumable && (
                <button
                  className="resume"
                  disabled={w.busy}
                  title={
                    w.busy
                      ? "任务正在使用该工作区，取消任务后再继续"
                      : "从断点继续处理该源视频，不重算已完成阶段"
                  }
                  onClick={() => handleResumeWorkspace(w)}
                >
                  继续
                </button>
              )}
              {w.openable && !hasCompletedTask && (
                <button
                  className="resume"
                  disabled={w.busy}
                  title={
                    w.busy
                      ? "任务正在使用该工作区，稍后再进入"
                      : "复用已完成的核心分析，进入擦除目标说话人选择（认人/转写会快速补跑）"
                  }
                  onClick={() => handleOpenWorkspace(w)}
                >
                  选择屏蔽人声
                </button>
              )}
              <button
                className="delete"
                disabled={w.busy}
                title={
                  w.busy
                    ? "任务正在使用该工作区，取消任务后再删除"
                    : `删除工作区，释放 ${w.size_label}`
                }
                onClick={() => handleDeleteWorkspace(w)}
              >
                删除
              </button>
            </div>
            <div className="workspace-path">{w.source_path ?? w.path}</div>
          </div>
          );
        })}
      </section>
    </main>
  );
}

function EraseForm({
  task,
  speakers,
  names,
}: {
  task: Task;
  speakers: TimelineSpeaker[];
  names: Record<string, string>;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [outputDir, setOutputDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function toggle(label: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  }

  async function submit() {
    if (busy) return;
    setError(null);
    if (selected.size === 0) {
      setError("请先勾选要擦除的说话人");
      return;
    }
    setBusy(true);
    try {
      const resp = await fetch(`/api/tasks/${task.id}/erase`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speakers: [...selected].sort(),
          output_dir: outputDir.trim() || null,
        }),
      });
      const data = await resp.json().catch(() => null);
      if (!resp.ok) {
        setError(data?.detail ?? "发起擦除失败");
        return;
      }
      setSelected(new Set());
    } catch {
      setError("无法连接后端服务");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="erase-form">
      <div className="erase-title">擦除目标说话人（可多选）</div>
      <div className="erase-speakers">
        {speakers.map((sp) => (
          <label key={sp.label} className="erase-speaker">
            <input
              type="checkbox"
              checked={selected.has(sp.label)}
              onChange={() => toggle(sp.label)}
            />
            <span
              className="dot"
              style={{ background: speakerColor(speakers, sp.label) }}
            />
            {names[sp.label] ?? sp.name ?? sp.label}
          </label>
        ))}
      </div>
      <div className="erase-row">
        <label htmlFor={`erase-dir-${task.id}`}>输出目录</label>
        <input
          id={`erase-dir-${task.id}`}
          type="text"
          placeholder={`默认与源视频同目录（${dirName(task.video_path) || "/"}）`}
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
        />
        <button type="button" className="primary" onClick={submit} disabled={busy}>
          {busy ? "发起中…" : "发起擦除"}
        </button>
      </div>
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}

function SpeakerTimeline({
  taskId,
  timeline,
  names,
  samplesBySpeaker,
  samplesHint,
  playing,
  playPos,
  onPlay,
  onDelete,
  onRename,
  onStoreVoiceprint,
  voiceprintBusy,
  onReassign,
  onSplit,
  onDiscard,
  correctionError,
}: {
  taskId: string;
  timeline: Timeline;
  names: Record<string, string>;
  samplesBySpeaker: Record<string, TranscriptSample[]>;
  samplesHint: string | null;
  playing: Playing | null;
  playPos: number | null;
  onPlay: (
    taskId: string,
    speaker: string,
    seg: Pick<TimelineSegment, "start" | "end">,
  ) => void;
  onDelete: (
    segment: TimelineSegment,
  ) => Promise<{ ok: boolean; detail: string | null }>;
  onRename: (label: string, name: string) => Promise<string | null | undefined>;
  onStoreVoiceprint: (label: string) => Promise<void>;
  voiceprintBusy: (label: string) => boolean;
  onReassign: (segment: TimelineSegment, target: string) => Promise<boolean>;
  onSplit: (
    segment: TimelineSegment,
    results: SplitResult[],
  ) => Promise<{ ok: boolean; detail: string | null; timeline?: Timeline }>;
  onDiscard: () => Promise<void>;
  correctionError: string | null;
}) {
  const [detail, setDetail] = useState<{
    speaker: string;
    segment: TimelineSegment;
  } | null>(null);
  const correction = timeline.correction;

  function openDetail(speaker?: string, segment?: TimelineSegment) {
    if (speaker && segment) {
      setDetail({ speaker, segment });
      return;
    }
    const first = firstSegment(timeline);
    if (first) setDetail(first);
  }

  return (
    <div className="timeline">
      <div className="timeline-title">
        说话人时间轴总览（共 {timeline.num_speakers} 人，点击总览打开校正详情；点击名字可命名）
      </div>
      {samplesHint && <p className="timeline-hint">{samplesHint}</p>}
      {correction.message && <p className="correction-warning">{correction.message}</p>}
      {correction.locked_reason && correction.state !== "invalidated" && (
        <p className="correction-warning">{correction.locked_reason}</p>
      )}
      {(correction.state === "active" || correction.state === "invalidated") && (
        <div className="correction-actions">
          <button
            type="button"
            onClick={() => onDiscard()}
            disabled={correction.state !== "invalidated" && !correction.editable}
          >
            {correction.state === "invalidated" ? "放弃作废校正" : "放弃全部校正"}
          </button>
        </div>
      )}
      {correctionError && <p className="error-text">{correctionError}</p>}
      <div className="timeline-canvas">
        {timeline.speakers.map((sp) => {
          const color = speakerColor(timeline.speakers, sp.label);
          const samples = samplesBySpeaker[sp.label] ?? [];
          return (
            <div key={sp.label} className="timeline-row">
              <span className="speaker-chip">
                <span className="dot" style={{ background: color }} />
                <SpeakerName
                  label={sp.label}
                  name={names[sp.label] ?? sp.name ?? null}
                  auto={Boolean(sp.auto && !names[sp.label])}
                  editable={correction.editable}
                  onRename={onRename}
                />
                <span className="speech-time">{fmtTime(sp.speech_time)}</span>
                <button
                  type="button"
                  className="store-voiceprint"
                  disabled={voiceprintBusy(sp.label)}
                  title={`从 ${shownName(names, sp)} 的当前生效时段抽取声纹并存入声纹库`}
                  onClick={() => void onStoreVoiceprint(sp.label)}
                >
                  {voiceprintBusy(sp.label) ? "存入中…" : "存入声纹库"}
                </button>
              </span>
              <SegmentTrack
                taskId={taskId}
                duration={timeline.duration}
                speaker={sp}
                color={color}
                playing={playing}
                selectedKey={null}
                onSelect={(speakerLabel, seg) => openDetail(speakerLabel, seg)}
                onBlankClick={() => openDetail()}
                variant="main"
              />
              {samples.length > 0 && (
                <div className="speaker-samples">
                  <span className="samples-label">样本台词</span>
                  <div className="sample-list">
                    {samples.map((s, i) => (
                      <button
                        key={i}
                        type="button"
                        className={`sample${
                          isSegmentActive(taskId, playing, sp.label, s) ? " playing" : ""
                        }`}
                        title={`试听 ${shownName(names, sp)} 的样本 ${fmtTime(s.start)} – ${fmtTime(s.end)}`}
                        onClick={() => onPlay(taskId, sp.label, s)}
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
              )}
            </div>
          );
        })}
        {timeline.duration > 0 && (
          <div className="timeline-axis">
            <span />
            <div className="timeline-axis-canvas">
              {Array.from({ length: 5 }, (_, i) => (
                <span key={i}>{fmtTime((timeline.duration * i) / 4)}</span>
              ))}
            </div>
          </div>
        )}
      </div>
      {detail && (
        <DetailModal
          taskId={taskId}
          timeline={timeline}
          names={names}
          playing={playing}
          playPos={playPos}
          editable={correction.editable}
          onPlay={onPlay}
          onDelete={onDelete}
          onReassign={onReassign}
          onSplit={onSplit}
          initial={detail}
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  );
}

// 总览轨道中的单个说话人时段绘制：整片时长映射到可用轨道宽度，短时段保留
// 1px 最小可视宽度，超过容量时允许相邻时段视觉覆盖。主界面与详情弹窗顶部
// 总览共用同一套绘制，只是样式类名不同。
function SegmentTrack({
  taskId,
  duration,
  speaker,
  color,
  playing,
  selectedKey,
  onSelect,
  onBlankClick,
  variant,
}: {
  taskId: string;
  duration: number;
  speaker: TimelineSpeaker;
  color: string;
  playing: Playing | null;
  selectedKey: string | null;
  onSelect?: (speakerLabel: string, seg: TimelineSegment) => void;
  onBlankClick?: () => void;
  variant: "main" | "overview";
}) {
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

// 校正详情弹窗：约 1000px 宽。顶部是全片总览（所有说话人、时间对齐、1px 最小
// 宽度、悬停显示固定 8 秒放大镜），点击总览中的时段选中它；下方是选中时段的
// 操作面板：试听、改派与完整切分（添加/拖动/删除间隔线、结果区间分配、保存）。
function DetailModal({
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
}: {
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
}) {
  const [selected, setSelected] = useState<{
    speaker: string;
    segment: TimelineSegment;
  } | null>(initial);
  const [draftDirty, setDraftDirty] = useState(false);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverClientX, setHoverClientX] = useState(0);
  const [hoverClientY, setHoverClientY] = useState(0);
  const [reassigning, setReassigning] = useState<TimelineSegment | null>(null);
  const [saving, setSaving] = useState(false);
  const overviewRef = useRef<HTMLDivElement | null>(null);

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

  // 固定 8 秒放大镜窗口：以悬停时间为中心，靠近片头/片尾时整体平移保持 8 秒。
  const MAGNIFIER_SECONDS = 8;
  const windowStart =
    duration > 0 && hoverTime !== null
      ? Math.min(
          Math.max(hoverTime - MAGNIFIER_SECONDS / 2, 0),
          Math.max(0, duration - MAGNIFIER_SECONDS),
        )
      : 0;
  const windowEnd =
    duration > 0 && hoverTime !== null
      ? Math.min(duration, windowStart + MAGNIFIER_SECONDS)
      : 0;
  const windowSeconds = Math.max(0, windowEnd - windowStart);
  const inWindow = (seg: TimelineSegment) =>
    seg.end > windowStart && seg.start < windowEnd;
  const windowLeft = (seg: TimelineSegment) =>
    Math.max(0, ((seg.start - windowStart) / windowSeconds) * 100);
  const windowRight = (seg: TimelineSegment) =>
    Math.min(100, ((seg.end - windowStart) / windowSeconds) * 100);
  const centerPct =
    windowSeconds > 0 && hoverTime !== null
      ? Math.min(100, Math.max(0, ((hoverTime - windowStart) / windowSeconds) * 100))
      : 50;

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
    let timePct = (event.clientX - rect.left) / rect.width;
    if (trackEl) {
      const trackRect = trackEl.getBoundingClientRect();
      if (trackRect.width > 0) {
        const trackX = Math.min(
          Math.max(event.clientX - trackRect.left, 0),
          trackRect.width,
        );
        timePct = trackX / trackRect.width;
      }
    }
    setHoverTime(Math.min(duration, Math.max(0, timePct) * duration));
  }

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

  // 放大镜固定悬浮在鼠标上方：left = 鼠标 x - centerPct% × 放大镜 padding box 宽
  // （360 - 左右边框 2px），使 hoverTime 的渲染位置（中线）始终对准鼠标。
  const magnifierWidth = 360;
  const magnifierPaddingBox = magnifierWidth - 2;
  const magnifierLeft = Math.min(
    Math.max(
      hoverClientX - (centerPct / 100) * magnifierPaddingBox,
      8,
    ),
    window.innerWidth - magnifierWidth - 8,
  );
  const magnifierTop = Math.max(hoverClientY, 140);

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
                  selectedKey={selected ? `${selected.speaker}:${selected.segment.index}` : null}
                  onSelect={handleSelect}
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
            <div
              className="magnifier"
              style={{ left: magnifierLeft, top: magnifierTop }}
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
                            left: `${windowLeft(seg).toFixed(3)}%`,
                            width: `${Math.max(0, windowRight(seg) - windowLeft(seg)).toFixed(3)}%`,
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
          )}
        </div>
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

// 选中时段操作面板：把选中时段整体拉宽展示，点击添加切分间隔线、拖动微调、
// 悬停间隔线时上方显示删除按钮；每个结果区间可分配给现有说话人或草稿内新建
// 槽位。切分草稿不参与生效时段，保存前完整校验并一次性替换原时段。
function SplitPanel({
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
}: {
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
}) {
  const start = segment.start;
  const end = segment.end;
  const [markers, setMarkers] = useState<number[]>([]);
  // 与结果区间一一对应的目标数组：现有说话人标签，或草稿新建槽位 id。
  const [targets, setTargets] = useState<string[]>([originalSpeaker]);
  const [slots, setSlots] = useState<string[]>([]);
  const [nextSlot, setNextSlot] = useState(1);
  const [hovered, setHovered] = useState<number | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [trackWidth, setTrackWidth] = useState(0);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const dragGuard = useRef(false);

  const intervals = splitIntervals(start, end, markers);
  const dirty =
    markers.length > 0 ||
    slots.length > 0 ||
    targets.some((t) => t !== originalSpeaker);
  useEffect(() => {
    onDirtyChange(dirty);
  }, [dirty, onDirtyChange]);

  const originalSpeakerInfo = timeline.speakers.find(
    (sp) => sp.label === originalSpeaker,
  );
  const pct = (t: number) =>
    `${(((t - start) / (end - start)) * 100).toFixed(3)}%`;
  // 切分线定位：按 track 像素取整，避免亚像素渲染导致线条粗细/明暗不一致。
  const markerLeft = (t: number): string =>
    trackWidth > 0
      ? `${Math.round(((t - start) / (end - start)) * trackWidth)}px`
      : pct(t);
  useLayoutEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const update = () => setTrackWidth(el.clientWidth);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // 结果区间宽度按「时长」占选中时段的比例换算，不能复用 pct（pct 的入参是绝对时间）。
  const durPct = (d: number) => `${((d / (end - start)) * 100).toFixed(3)}%`;

  function timeFromClientX(clientX: number): number | null {
    const el = trackRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    return start + frac * (end - start);
  }

  function addMarker(time: number) {
    if (!editable || time <= start || time >= end) return;
    const next = [...markers, time].sort((a, b) => a - b);
    const points = [start, ...next, end];
    for (let i = 1; i < points.length - 1; i++) {
      if (points[i] - points[i - 1] < MIN_SPLIT_SECONDS - SPLIT_EPS) {
        setError("间隔线间距不足 0.1 秒，无法在该位置添加");
        return;
      }
    }
    const insertAt = intervals.findIndex((iv) => time <= iv.end + SPLIT_EPS);
    const idx = insertAt < 0 ? intervals.length - 1 : insertAt;
    const newTargets = [...targets];
    // 新区间继承被切开区间的目标
    newTargets.splice(idx + 1, 0, targets[idx] ?? originalSpeaker);
    setMarkers(next);
    setTargets(newTargets);
    setError(null);
  }

  function deleteMarker(markerIndex: number) {
    if (!editable) return;
    setMarkers((prev) => prev.filter((_, i) => i !== markerIndex));
    setTargets((prev) => {
      const next = [...prev];
      // 合并相邻两个区间，保留左侧目标
      next.splice(markerIndex + 1, 1);
      return next;
    });
    setHovered(null);
    setError(null);
  }

  function discardDraft() {
    setMarkers([]);
    setTargets([originalSpeaker]);
    setSlots([]);
    setNextSlot(1);
    setError(null);
  }

  function addSlot() {
    if (!editable) return;
    setSlots((prev) => [...prev, `new-slot-${nextSlot}`]);
    setNextSlot((n) => n + 1);
    setError(null);
  }

  function markerBounds(index: number): { lower: number; upper: number } {
    const prev = index > 0 ? markers[index - 1] : start;
    const nextBound = index < markers.length - 1 ? markers[index + 1] : end;
    return { lower: prev + MIN_SPLIT_SECONDS, upper: nextBound - MIN_SPLIT_SECONDS };
  }

  function onMarkerPointerDown(
    event: ReactPointerEvent<HTMLDivElement>,
    index: number,
  ) {
    if (!editable) return;
    event.preventDefault();
    event.stopPropagation();
    dragGuard.current = true;
    setDragging(index);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onMarkerPointerMove(
    event: ReactPointerEvent<HTMLDivElement>,
    index: number,
  ) {
    if (dragging !== index) return;
    const time = timeFromClientX(event.clientX);
    if (time === null) return;
    const { lower, upper } = markerBounds(index);
    const clamped = Math.min(Math.max(time, lower), upper);
    if (Math.abs(clamped - markers[index]) > SPLIT_EPS) {
      dragGuard.current = true;
    }
    setMarkers((prev) => {
      const next = [...prev];
      next[index] = clamped;
      return next;
    });
    setError(null);
  }

  function onMarkerPointerUp(
    event: ReactPointerEvent<HTMLDivElement>,
    index: number,
  ) {
    if (dragging !== index) return;
    setDragging(null);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // 指针捕获已释放时忽略
    }
  }

  function onTrackClick(event: ReactMouseEvent<HTMLDivElement>) {
    if (!editable) return;
    if (dragGuard.current) {
      dragGuard.current = false;
      return;
    }
    const time = timeFromClientX(event.clientX);
    if (time !== null) addMarker(time);
  }

  function validate(): string | null {
    // 切分必须至少两个结果区间；整段改说话人只走改派对话框。
    if (intervals.length < 2) {
      return "至少需要两个结果区间才能保存";
    }
    for (const iv of intervals) {
      if (iv.end - iv.start < MIN_SPLIT_SECONDS - SPLIT_EPS) {
        return "每个结果区间至少 0.1 秒";
      }
    }
    for (let i = 0; i < intervals.length; i++) {
      if (!targets[i]) return "所有结果区间都必须指定说话人";
    }
    if (intervals.every((_, i) => targets[i] === originalSpeaker)) {
      return "至少一个结果区间需要改派为非原说话人才能保存";
    }
    return null;
  }

  async function save() {
    const problem = validate();
    if (problem) {
      setError(problem);
      return;
    }
    const results: SplitResult[] = intervals.map((iv, i) => {
      const target = targets[i];
      if (isNewSlotTarget(target)) {
        return { start: iv.start, end: iv.end, target: NEW_SLOT_TOKEN, slot: target };
      }
      return { start: iv.start, end: iv.end, target };
    });
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
    setMarkers([]);
    setTargets([originalSpeaker]);
    setSlots([]);
    setNextSlot(1);
    setError(null);
  }

  function targetLabel(target: string): string {
    if (isNewSlotTarget(target)) {
      const n = slots.indexOf(target);
      return n >= 0 ? `新建说话人 ${n + 1}` : target;
    }
    return target;
  }

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
  const playBounds = (() => {
    if (
      playing === null ||
      playing.taskId !== taskId ||
      playing.speaker !== originalSpeaker
    ) {
      return null;
    }
    if (playing.start === start && playing.end === end) {
      return { from: start, to: end };
    }
    const iv = intervals.find(
      (x) => x.start === playing.start && x.end === playing.end,
    );
    return iv ? { from: iv.start, to: iv.end } : null;
  })();
  const playheadPos =
    playBounds !== null && playPos !== null
      ? Math.min(playBounds.to, Math.max(playBounds.from, playPos))
      : null;

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

// 改派说话人时段的确认弹窗（在详情弹窗内复用原校正对话框样式）。
function ReassignDialog({
  segment,
  timeline,
  names,
  saving,
  onCancel,
  onSave,
}: {
  segment: TimelineSegment;
  timeline: Timeline;
  names: Record<string, string>;
  saving: boolean;
  onCancel: () => void;
  onSave: (target: string) => void;
}) {
  const [target, setTarget] = useState("__new__");
  return (
    <div
      className="correction-dialog-backdrop"
      role="presentation"
      onClick={onCancel}
    >
      <div
        className="correction-dialog"
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <h3>改派说话人时段</h3>
        <p className="correction-dialog-meta">
          {fmtTime(segment.start)} – {fmtTime(segment.end)}
        </p>
        <label>
          新说话人
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="__new__">新建说话人（顺延编号）</option>
            {timeline.speakers.map((sp) => (
              <option key={sp.label} value={sp.label}>
                {shownName(names, sp)}
              </option>
            ))}
          </select>
        </label>
        <div className="correction-dialog-actions">
          <button type="button" onClick={onCancel}>
            取消
          </button>
          <button
            type="button"
            className="primary"
            disabled={saving}
            onClick={() => onSave(target)}
          >
            {saving ? "保存中…" : "保存改派"}
          </button>
        </div>
      </div>
    </div>
  );
}

function SpeakerName({
  label,
  name,
  auto,
  editable,
  onRename,
}: {
  label: string;
  name: string | null;
  auto?: boolean;
  editable?: boolean;
  onRename: (label: string, name: string) => Promise<string | null | undefined>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  async function commit() {
    if (busy) return;
    setBusy(true);
    const final = await onRename(label, draft);
    if (final === undefined) {
      setBusy(false);  // 失败：保持编辑态，草稿保留供重试或 Esc 取消
      return;
    }
    setDraft(final ?? "");
    setBusy(false);
    setEditing(false);
  }

  if (!editing) {
    return (
      <span className="speaker-name-wrap">
        <button
          type="button"
          className="speaker-name"
          disabled={editable === false}
          title={name ? `改名「${name}」` : `为 ${label} 命名`}
          onClick={() => {
            setDraft(name ?? "");
            setEditing(true);
          }}
        >
          {name ?? label}
        </button>
        {auto && (
          <span
            className="speaker-auto"
            title="自动认人结果，手工命名会覆盖它"
          >
            自动
          </span>
        )}
      </span>
    );
  }
  return (
    <input
      className="speaker-name-input"
      value={draft}
      autoFocus
      maxLength={100}
      placeholder={label}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        else if (e.key === "Escape") setEditing(false);
      }}
    />
  );
}
