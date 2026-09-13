// 声纹库校验面板：进度、终态说明与逐条建议的接受/忽略。
import { effectiveCheck, suggestionText } from "../lib/voiceprint";
import type { CheckActions, Task, VoiceprintCheckState } from "../types";

export interface VoiceprintCheckPanelProps {
  task: Task;
  checkStates: Record<string, VoiceprintCheckState>;
  checks: CheckActions;
}

export function VoiceprintCheckPanel({
  task,
  checkStates,
  checks,
}: VoiceprintCheckPanelProps) {
  const check = effectiveCheck(task, checkStates);
  if (!check) return null;
  const active = check.state === "queued" || check.state === "running";
  return (
    <div className="voiceprint-check">
      <div className="voiceprint-check-head">
        <strong>声纹库校验</strong>
        {active ? (
          <>
            <span>{Math.round((check.progress ?? 0) * 100)}%</span>
            <button className="cancel" onClick={() => checks.cancelCheck(task)}>取消</button>
          </>
        ) : (
          <button className="resume" onClick={() => checks.startCheck(task)}>用声纹库校验</button>
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
                    disabled={checks.isDecisionBusy(task, i)}
                    onClick={() => checks.decideSuggestion(task, i, "accept")}
                  >
                    接受
                  </button>
                  <button
                    type="button"
                    className="cancel"
                    disabled={checks.isDecisionBusy(task, i)}
                    onClick={() => checks.decideSuggestion(task, i, "ignore")}
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
