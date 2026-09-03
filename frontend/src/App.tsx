import { useCallback, useEffect, useState } from "react";

interface Stage {
  id: string;
  name: string;
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  error: string | null;
  artifacts: string[];
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

export default function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [videoPath, setVideoPath] = useState("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [selectedTrack, setSelectedTrack] = useState(0);
  const [formError, setFormError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const resp = await fetch("/api/tasks");
      const data = await resp.json();
      setTasks(data.tasks);
    } catch {
      // 后端暂不可达时保留旧列表，下一轮重试
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [refresh]);

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
                    </span>
                  ))}
                </span>
              </div>
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
