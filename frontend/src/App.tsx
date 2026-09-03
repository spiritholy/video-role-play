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

interface ProbeResult {
  container: string;
  duration: number;
  tracks: ProbeTrack[];
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
  if (task.state === "completed") return "全部阶段完成";
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
  const [timelines, setTimelines] = useState<Record<string, TimelineOrError>>({});
  // 说话人命名（工作区元数据的本地镜像）：工作区 → 编号标签 → 名字。
  // 按工作区键控：命名是工作区级数据，同源多任务卡片共享同一份显示。
  const [speakerNames, setSpeakerNames] = useState<
    Record<string, Record<string, string>>
  >({});
  const [transcripts, setTranscripts] = useState<Record<string, TranscriptsOrError>>({});
  const [playing, setPlaying] = useState<Playing | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const requestedTimelines = useRef<Set<string>>(new Set());
  const requestedTranscripts = useRef<Set<string>>(new Set());
  const [videoPath, setVideoPath] = useState("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [selectedTrack, setSelectedTrack] = useState(0);
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
        setSpeakerNames((prev) => ({
          ...prev,
          [wsKey]: Object.fromEntries(
            data.speakers.filter((s) => s.name).map((s) => [s.label, s.name as string]),
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
  }, [maybeFetchTimeline, maybeFetchTranscripts]);

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

    return (
      <SpeakerTimeline
        taskId={t.id}
        timeline={timeline}
        names={speakerNames[workspaceKey(t)] ?? {}}
        samplesBySpeaker={samplesBySpeaker}
        samplesHint={samplesHint}
        playing={playing}
        onPlay={playSegment}
        onRename={(label, name) => renameSpeaker(t, label, name)}
      />
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

        {formError && <p className="error-text">{formError}</p>}

        <button type="submit" className="primary">
          提交处理任务
        </button>
      </form>

      <section className="tasks">
        <h2>处理任务</h2>
        {tasks.length === 0 && <p className="empty">还没有任务，提交一个源视频开始。</p>}
        {tasks.map((t) => {
          const cancellable = t.state === "queued" || t.state === "running";
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
                      {s.status === "running" && s.progress != null &&
                        ` ${Math.round(s.progress * 100)}%`}
                    </span>
                  ))}
                </span>
              </div>
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
    </main>
  );
}

function fileName(path: string): string {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return i >= 0 ? path.slice(i + 1) : path;
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
}: {
  taskId: string;
  timeline: Timeline;
  names: Record<string, string>;
  samplesBySpeaker: Record<string, TranscriptSample[]>;
  samplesHint: string | null;
  playing: Playing | null;
  onPlay: (taskId: string, speaker: string, seg: TimelineSegment) => void;
  onRename: (label: string, name: string) => Promise<string | null | undefined>;
}) {
  const pct = (v: number) => `${((v / timeline.duration) * 100).toFixed(3)}%`;
  const shown = (label: string) => names[label] ?? label;
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
                name={names[sp.label] ?? null}
                onRename={onRename}
              />
              <span className="speech-time">{fmtTime(sp.speech_time)}</span>
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
                  title={`${shown(sp.label)} · ${fmtTime(seg.start)} – ${fmtTime(seg.end)}`}
                  aria-label={`试听 ${shown(sp.label)}，${fmtTime(seg.start)} 到 ${fmtTime(seg.end)}`}
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
                      title={`试听 ${shown(sp.label)} 的样本 ${fmtTime(s.start)} – ${fmtTime(s.end)}`}
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
  onRename,
}: {
  label: string;
  name: string | null;
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
