// 阶段清单 (Stage Manifest) 的进度 chips：运行中的阶段内嵌进度条。
import type { Stage } from "../types";

export interface StageChipsProps {
  stages: Stage[];
}

export function StageChips({ stages }: StageChipsProps) {
  return (
    <span className="stage-chips">
      {stages.map((s) => (
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
  );
}
