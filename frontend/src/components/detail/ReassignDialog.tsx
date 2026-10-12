// 改派说话人对话框：整段改派给现有说话人或新建说话人。
import { useState } from "react";
import { fmtTime } from "../../lib/format";
import { shownName } from "../../lib/speakers";
import type { Timeline, TimelineSegment } from "../../types";

export interface ReassignDialogProps {
  segment: TimelineSegment;
  timeline: Timeline;
  names: Record<string, string>;
  saving: boolean;
  onCancel: () => void;
  onSave: (target: string) => void;
}

// 改派说话人时段的确认弹窗（在详情弹窗内复用原校正对话框样式）。
export function ReassignDialog({
  segment,
  timeline,
  names,
  saving,
  onCancel,
  onSave,
}: ReassignDialogProps) {
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
