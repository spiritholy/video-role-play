// 已完成阶段的产物试听：人声轨 / 伴奏轨等分轨音频。
import { audioUrl } from "../api/urls";
import { STEM_LABELS } from "../lib/labels";
import type { Stage } from "../types";

export interface StemPlayersProps {
  taskId: string;
  stages: Stage[];
}

export function StemPlayers({ taskId, stages }: StemPlayersProps) {
  return (
    <>
      {stages
        .filter((s) => s.status === "done" && Object.keys(s.outputs).length > 0)
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
                  src={audioUrl(taskId, artifact)}
                />
              </div>
            ))}
          </div>
        ))}
    </>
  );
}
