// 说话人配色与展示名。
import type { TimelineSpeaker } from "../types";

export const SPEAKER_COLORS = [
  "#4f7cff", "#e0533d", "#2ba24c", "#9d5bd2",
  "#e8a13a", "#2ab5b5", "#d45791", "#7a8699",
];

export function speakerColor(speakers: TimelineSpeaker[], label: string): string {
  const i = speakers.findIndex((s) => s.label === label);
  return SPEAKER_COLORS[i % SPEAKER_COLORS.length];
}

export function shownName(names: Record<string, string>, sp: TimelineSpeaker): string {
  return names[sp.label] ?? sp.name ?? sp.label;
}
