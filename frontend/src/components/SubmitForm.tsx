// 提交处理任务表单：源视频路径、音轨探测结果与说话人数量提示。
import { trackLabel } from "../lib/format";
import type { ProbeResult } from "../types";

export interface SubmitFormProps {
  videoPath: string;
  setVideoPath: (value: string) => void;
  probe: ProbeResult | null;
  selectedTrack: number;
  setSelectedTrack: (value: number) => void;
  numSpeakers: string;
  setNumSpeakers: (value: string) => void;
  formError: string | null;
  probing: boolean;
  handleProbe: () => Promise<void>;
  handleSubmit: (e: React.FormEvent) => Promise<void>;
}

export function SubmitForm({
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
}: SubmitFormProps) {
  return (
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
  );
}
