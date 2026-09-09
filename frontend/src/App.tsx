import { useCallback, useEffect, useRef, useState } from "react";

interface Stage {
  id: string;
  name: string;
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  error: string | null;
  artifacts: string[];
  progress: number | null;
  outputs: Record<string, string>;
}

const STEM_LABELS: Record<string, string> = {
  vocal: "人声轨",
  accompaniment: "伴奏轨",
};

interface TimelineSegment {
  start: number;
  end: number;
}

interface TimelineSpeaker {
  label: string;
  name: string | null;
  auto?: boolean;  // 名字是否自动认出来的（手工命名优先，不标自动）
  speech_time: number;
  segments: TimelineSegment[];
}

interface Timeline {
  model: string;
  num_speakers: number;
  duration: number;
  speakers: TimelineSpeaker[];
}

type TimelineOrError = Timeline | "error";

interface TranscriptSample {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

interface Transcripts {
  model: string;
  samples: TranscriptSample[];
}

type TranscriptsOrError = Transcripts | "error";

interface Playing {
  taskId: string;
  speaker: string;
  start: number;
  end: number;
}

const SPEAKER_COLORS = [
  "#4f7cff", "#e0533d", "#2ba24c", "#9d5bd2",
  "#e8a13a", "#2ab5b5", "#d45791", "#7a8699",
];

function speakerColor(speakers: TimelineSpeaker[], label: string): string {
  const i = speakers.findIndex((s) => s.label === label);
  return SPEAKER_COLORS[i % SPEAKER_COLORS.length];
}

function fmtTime(t: number): string {
  if (t < 60) return `${t.toFixed(1)}s`;
  const m = Math.floor(t / 60);
  const s = Math.round(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

interface Task {
  id: string;
  video_path: string;
  audio_track: number | null;
  state: "queued" | "running" | "completed" | "failed" | "cancelled";
  state_label: string;
  error: string | null;
  stages: Stage[];
  output_path: string | null;
  workspace_dir: string | null;
  erase: { speakers: string[]; output_dir: string | null } | null;
  created_at: string;
}

interface ProbeTrack {
  index: number;
  stream_index: number;
  codec: string | null;
  channels: number | null;
  channel_layout: string | null;
  language: string | null;
}

interface WorkspaceItem {
  name: string;
  path: string;
  source_path: string | null;
  size_bytes: number;
  size_label: string;
  reusable: boolean;
  openable: boolean;
  resumable: boolean;
  busy: boolean;
}

interface WorkspaceList {
  workspaces: WorkspaceItem[];
  total_bytes: number;
  total_label: string;
}

interface ProbeResult {
  container: string;
  duration: number;
  tracks: ProbeTrack[];
}

interface VoiceprintName {
  name: string;
  vectors_count: number;
  source_workspace: string | null;
  updated_at: string | null;
}

interface VoiceprintList {
  revision: number;
  model: string;
  names: VoiceprintName[];
}

function trackLabel(t: ProbeTrack): string {
  const parts = [`音轨 ${t.index + 1}`, t.codec ?? "未知编码"];
  if (t.channels != null) {
    parts.push(
      t.channel_layout ? `${t.channel_layout}（${t.channels} 声道）` : `${t.channels} 声道`
    );
  }
  if (t.language) parts.push(`语言 ${t.language}`);
  return parts.join(" · ");
}

function stageSummary(task: Task): string {
  if (task.state === "completed") {
    const eraseStage = task.stages.find((s) => s.id === "encode_mux");
    return eraseStage?.status === "done" ? "擦除完成" : "分析完成，可选择要擦除的说话人";
  }
  const idx = task.stages.findIndex((s) => s.status === "running");
  if (idx >= 0) return `阶段 ${idx + 1}/${task.stages.length} · ${task.stages[idx].name}`;
  const firstUndone = task.stages.findIndex(
    (s) => s.status !== "done" && s.status !== "cancelled"
  );
  if (firstUndone >= 0) return `阶段 ${firstUndone + 1}/${task.stages.length} · ${task.stages[firstUndone].name}`;
  return "";
}

function audioUrl(taskId: string, artifact: string): string {
  return `/api/tasks/${taskId}/audio/${
    artifact.split("/").map(encodeURIComponent).join("/")
  }`;
}

function workspaceKey(t: Task): string {
  return t.workspace_dir ?? t.id;
}

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
  const [playing, setPlaying] = useState<Playing | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const requestedTimelines = useRef<Set<string>>(new Set());
  const requestedTranscripts = useRef<Set<string>>(new Set());
  const [videoPath, setVideoPath] = useState("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [selectedTrack, setSelectedTrack] = useState(0);
  const [numSpeakers, setNumSpeakers] = useState("");  // 空=让模型自动估计
  const [formError, setFormError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);

  const maybeFetchTimeline = useCallback(
    async (t: Task) => {
      if (!t.stages.some((s) => s.id === "diarize" && s.status === "done")) return;
      if (requestedTimelines.current.has(t.id)) return;
      requestedTimelines.current.add(t.id);
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
    } catch {
      // 后端暂不可达时保留旧列表，下一轮重试
    }
    refreshWorkspaces();
    refreshVoiceprints();
  }, [maybeFetchTimeline, maybeFetchTranscripts, refreshWorkspaces, refreshVoiceprints]);

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

  function playSegment(taskId: string, speaker: string, seg: TimelineSegment) {
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
          onPlay={playSegment}
          onRename={(label, name) => renameSpeaker(t, label, name)}
          onStoreVoiceprint={(label) => handleStoreVoiceprint(t, label)}
          voiceprintBusy={(rowLabel) => Boolean(voiceprintBusy[`${t.id}:${rowLabel}`])}
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
            在说话人列表中显式存入声纹，后续剧集据此自动认人/纠错（带「自动」标记，
            手工命名永远优先）。同名人物再次存入会追加声纹。误入库的人物可在此删除。
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

function fileName(path: string): string {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return i >= 0 ? path.slice(i + 1) : path;
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

function dirName(path: string): string {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return i >= 0 ? path.slice(0, i) : "";
}

function SpeakerTimeline({
  taskId,
  timeline,
  names,
  samplesBySpeaker,
  samplesHint,
  playing,
  onPlay,
  onRename,
  onStoreVoiceprint,
  voiceprintBusy,
}: {
  taskId: string;
  timeline: Timeline;
  names: Record<string, string>;
  samplesBySpeaker: Record<string, TranscriptSample[]>;
  samplesHint: string | null;
  playing: Playing | null;
  onPlay: (taskId: string, speaker: string, seg: TimelineSegment) => void;
  onRename: (label: string, name: string) => Promise<string | null | undefined>;
  onStoreVoiceprint: (label: string) => Promise<void>;
  voiceprintBusy: (label: string) => boolean;
}) {
  const pct = (v: number) => `${((v / timeline.duration) * 100).toFixed(3)}%`;
  const shown = (sp: TimelineSpeaker) => names[sp.label] ?? sp.name ?? sp.label;
  const isActive = (sp: TimelineSpeaker, seg: TimelineSegment) =>
    playing !== null &&
    playing.taskId === taskId &&
    playing.speaker === sp.label &&
    playing.start === seg.start &&
    playing.end === seg.end;

  return (
    <div className="timeline">
      <div className="timeline-title">
        说话人时间轴（共 {timeline.num_speakers} 人，点击名字可命名，点击任意时段试听）
      </div>
      {samplesHint && <p className="timeline-hint">{samplesHint}</p>}
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
                onRename={onRename}
              />
              <span className="speech-time">{fmtTime(sp.speech_time)}</span>
              <button
                type="button"
                className="store-voiceprint"
                disabled={voiceprintBusy(sp.label)}
                title={`从 ${shown(sp)} 的当前生效时段抽取声纹并存入声纹库`}
                onClick={() => void onStoreVoiceprint(sp.label)}
              >
                {voiceprintBusy(sp.label) ? "存入中…" : "存入声纹库"}
              </button>
            </span>
            <div className="timeline-track">
              {sp.segments.map((seg, i) => (
                <button
                  key={i}
                  type="button"
                  className={`timeline-seg${isActive(sp, seg) ? " playing" : ""}`}
                  style={{
                    left: pct(seg.start),
                    width: pct(seg.end - seg.start),
                    background: color,
                  }}
                  title={`${shown(sp)} · ${fmtTime(seg.start)} – ${fmtTime(seg.end)}`}
                  aria-label={`试听 ${shown(sp)}，${fmtTime(seg.start)} 到 ${fmtTime(seg.end)}`}
                  onClick={() => onPlay(taskId, sp.label, seg)}
                />
              ))}
            </div>
            {samples.length > 0 && (
              <div className="speaker-samples">
                <span className="samples-label">样本台词</span>
                <div className="sample-list">
                  {samples.map((s, i) => (
                    <button
                      key={i}
                      type="button"
                      className={`sample${isActive(sp, s) ? " playing" : ""}`}
                      title={`试听 ${shown(sp)} 的样本 ${fmtTime(s.start)} – ${fmtTime(s.end)}`}
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
          <div>
            {Array.from({ length: 5 }, (_, i) => (
              <span key={i}>{fmtTime((timeline.duration * i) / 4)}</span>
            ))}
          </div>
          <span />
        </div>
      )}
    </div>
  );
}

function SpeakerName({
  label,
  name,
  auto,
  onRename,
}: {
  label: string;
  name: string | null;
  auto?: boolean;
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
