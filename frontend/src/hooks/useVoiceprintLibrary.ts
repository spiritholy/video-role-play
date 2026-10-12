// 人物声纹库的删除：删掉后后续剧集不再自动识别该人物。
import { useCallback, useState } from "react";
import { errorMessage } from "../api/client";
import { deleteVoiceprintEntry } from "../api/voiceprints";

export interface VoiceprintLibrary {
  voiceprintError: string | null;
  handleDeleteVoiceprint: (name: string) => Promise<void>;
}

export function useVoiceprintLibrary(deps: {
  refreshVoiceprints: () => Promise<void>;
}): VoiceprintLibrary {
  const { refreshVoiceprints } = deps;
  const [voiceprintError, setVoiceprintError] = useState<string | null>(null);

  const handleDeleteVoiceprint = useCallback(
    async (name: string) => {
      const confirmed = window.confirm(
        `从声纹库删除「${name}」？\n\n后续剧集将不再自动识别该人物。`,
      );
      if (!confirmed) return;
      setVoiceprintError(null);
      const result = await deleteVoiceprintEntry(name);
      if (!result.ok) {
        setVoiceprintError(errorMessage(result, "删除声纹失败"));
        return;
      }
      refreshVoiceprints();
    },
    [refreshVoiceprints],
  );

  return { voiceprintError, handleDeleteVoiceprint };
}
