// 擦除目标说话人表单：多选说话人 + 可选输出目录，提交后排入擦除任务。
import { useState } from "react";
import { errorMessage } from "../api/client";
import { startErase } from "../api/tasks";
import { dirName } from "../lib/format";
import { speakerColor } from "../lib/speakers";
import type { Task, TimelineSpeaker } from "../types";

export interface EraseFormProps {
  task: Task;
  speakers: TimelineSpeaker[];
  names: Record<string, string>;
}

export function EraseForm({
  task,
  speakers,
  names,
}: EraseFormProps) {
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
    const result = await startErase(
      task.id,
      [...selected].sort(),
      outputDir.trim() || null,
    );
    setBusy(false);
    if (!result.ok) {
      setError(errorMessage(result, "发起擦除失败"));
      return;
    }
    setSelected(new Set());
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
