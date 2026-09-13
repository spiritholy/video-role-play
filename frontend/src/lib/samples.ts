// 样本台词 (Sample Segment) 的加载提示与按说话人分组。
import type { Stage, TranscriptSample, TranscriptsOrError } from "../types";

export interface DerivedSamples {
  samplesHint: string | null;
  samplesBySpeaker: Record<string, TranscriptSample[]>;
}

export function deriveSamples(
  data: TranscriptsOrError | undefined,
  transcribeStage: Stage | undefined,
): DerivedSamples {
  let samplesHint: string | null = null;
  const samplesBySpeaker: Record<string, TranscriptSample[]> = {};
  if (data === undefined) {
    if (
      transcribeStage &&
      (transcribeStage.status === "pending" ||
        transcribeStage.status === "running")
    ) {
      samplesHint = "样本转写中…";
    } else if (transcribeStage?.status === "done") {
      samplesHint = "正在加载样本台词…";
    }
  } else if (data === "error") {
    samplesHint = "样本台词加载失败";
  } else {
    for (const s of data.samples) {
      (samplesBySpeaker[s.speaker] ??= []).push(s);
    }
  }
  return { samplesHint, samplesBySpeaker };
}
