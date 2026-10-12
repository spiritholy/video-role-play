// 人物声纹库面板：库版本、已积累人物与逐个删除。
import type { VoiceprintList } from "../types";

export interface VoiceprintLibraryPanelProps {
  voiceprints: VoiceprintList | null;
  voiceprintError: string | null;
  onDeleteVoiceprint: (name: string) => Promise<void>;
}

export function VoiceprintLibraryPanel({
  voiceprints,
  voiceprintError,
  onDeleteVoiceprint,
}: VoiceprintLibraryPanelProps) {
  return (
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
            在说话人列表中显式存入声纹；跨视频复用需点击「用声纹库校验」。
            校验只给出命名、合并、改派建议，经你确认后才会修改。误入库的人物可在此删除。
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
                onClick={() => onDeleteVoiceprint(n.name)}
              >
                删除
              </button>
            </div>
          ))}
        </div>
  );
}
