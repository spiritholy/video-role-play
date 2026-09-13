// 说话人名字：点击进入行内编辑，Enter/失焦提交、Esc 取消，失败时保持编辑态。
import { useState } from "react";

export interface SpeakerNameProps {
  label: string;
  name: string | null;
  auto?: boolean;
  editable?: boolean;
  onRename: (label: string, name: string) => Promise<string | null | undefined>;
}

export function SpeakerName({
  label,
  name,
  auto,
  editable,
  onRename,
}: SpeakerNameProps) {
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
