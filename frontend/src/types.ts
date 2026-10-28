// 前端领域类型：与后端接口的响应结构一一对应。
// 术语口径见仓库根 GLOSSARY.md（说话人时段 / 时段切分 / 人工校正 / 声纹校验 …）。

/* ---------- 处理任务 (Task) 与阶段清单 (Stage Manifest) ---------- */
export interface Task {
  id: string;
  video_path: string;
  audio_track: number | null;
  state: "queued" | "running" | "completed" | "failed" | "cancelled";
  state_label: string;
  error: string | null;
  stages: Stage[];
  output_path: string | null;
  workspace_dir: string | null;
  erase: { speakers: string[]; output_dir: string | null } | null;
  voiceprint_check?: VoiceprintCheckState | null;
  created_at: string;
}

export interface Stage {
  id: string;
  name: string;
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  error: string | null;
  artifacts: string[];
  progress: number | null;
  outputs: Record<string, string>;
}

/* ---------- 说话人时段 (Speaker Segment) 与时间轴 ---------- */
export interface TimelineSegment {
  start: number;
  end: number;
  index: number;
}

export interface CorrectionStatus {
  state: "none" | "active" | "invalidated";
  revision: number;
  message: string | null;
  editable: boolean;
  locked_reason: string | null;
}

export interface TimelineSpeaker {
  label: string;
  name: string | null;
  auto?: boolean;  // 名字是否自动认出来的（手工命名优先，不标自动）
  speech_time: number;
  segments: TimelineSegment[];
}

export interface Timeline {
  model: string;
  num_speakers: number;
  duration: number;
  speakers: TimelineSpeaker[];
  correction: CorrectionStatus;
}

export type TimelineOrError = Timeline | "error";

export interface TimelineUpdate {
  taskId: string;
  workspaceKey: string;
  timeline: Timeline;
}

/* ---------- 样本段 (Sample Segment) 与台词转写 ---------- */
export interface TranscriptSample {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

export interface Transcripts {
  model: string;
  samples: TranscriptSample[];
}

export type TranscriptsOrError = Transcripts | "error";

/* ---------- 人工校正 (Manual Correction)：时段切分 ---------- */
export interface SplitInterval {
  key: string;
  start: number;
  end: number;
}

// 保存切分时的结果区间协议：target 为现有说话人标签；"__new__" 表示新建
// 说话人，slot 是草稿内的槽位标识（同一槽位后端映射为同一个 SPEAKER_XX）。
export interface SplitResult {
  start: number;
  end: number;
  target: string;
  slot?: string;
}

/* ---------- 声纹库 (Voiceprint Library) 与声纹校验 (Voiceprint Check) ---------- */
export interface VoiceprintSuggestion {
  type: "rename" | "merge" | "reassign";
  decision?: "pending" | "accepted" | "ignored";
  label?: string;
  name?: string;
  similarity?: number;
  labels?: string[];
  canonical?: string;
  from_label?: string;
  to_label?: string;
  start?: number;
  end?: number;
}

export interface VoiceprintCheckResult {
  status?: "pending" | "invalidated";
  suggestions: VoiceprintSuggestion[];
}

export interface VoiceprintCheckState {
  state: "not_run" | "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  error: string | null;
  result?: VoiceprintCheckResult | null;
}

export interface VoiceprintName {
  name: string;
  vectors_count: number;
  source_workspace: string | null;
  updated_at: string | null;
}

export interface VoiceprintList {
  revision: number;
  model: string;
  names: VoiceprintName[];
}

/* ---------- 工作区 (Workspace) 与音轨探测 ---------- */
export interface ProbeTrack {
  index: number;
  stream_index: number;
  codec: string | null;
  channels: number | null;
  channel_layout: string | null;
  language: string | null;
}

export interface WorkspaceItem {
  name: string;
  path: string;
  source_path: string | null;
  size_bytes: number;
  size_label: string;
  reusable: boolean;
  openable: boolean;
  resumable: boolean;
  busy: boolean;
}

export interface WorkspaceList {
  workspaces: WorkspaceItem[];
  total_bytes: number;
  total_label: string;
}

export interface ProbeResult {
  container: string;
  duration: number;
  tracks: ProbeTrack[];
}

/* ---------- 试听播放态 ---------- */
export interface Playing {
  taskId: string;
  speaker: string;
  start: number;
  end: number;
}
/* ---------- 人工校正写操作的结果（沿用拆分前的返回形状，不改语义） ---------- */

// 删除时段 / 放弃校正等只关心成败与错误文案的操作。
export type MutationOutcome = { ok: boolean; detail: string | null };

// 保存切分：成功时带回最新时间轴，供弹窗把选中段更新到保存后的首段。
export type SplitOutcome = MutationOutcome & { timeline?: Timeline };

/* ---------- hooks 暴露给组件的回调束（签名 = 拆分前 App 内对应函数） ---------- */

// 人工校正的三项段级写操作：改派、切分、时段删除，加上放弃全部校正。
export interface CorrectionActions {
  reassignSegment(
    task: Task,
    segment: TimelineSegment,
    target: string,
  ): Promise<boolean>;
  splitSegment(
    task: Task,
    segment: TimelineSegment,
    results: SplitResult[],
  ): Promise<SplitOutcome>;
  deleteSegment(task: Task, segment: TimelineSegment): Promise<MutationOutcome>;
  discardCorrections(task: Task): Promise<void>;
}

// 说话人命名与显式存入声纹库。
export interface NamingActions {
  renameSpeaker(
    task: Task,
    label: string,
    rawName: string,
  ): Promise<string | null | undefined>;
  storeVoiceprint(task: Task, label: string): Promise<void>;
  isVoiceprintBusy(task: Task, label: string): boolean;
}

// 按需声纹校验：启动、取消、逐条裁决建议。
export interface CheckActions {
  startCheck(task: Task): Promise<void>;
  cancelCheck(task: Task): Promise<void>;
  decideSuggestion(
    task: Task,
    index: number,
    decision: "accept" | "ignore",
  ): Promise<void>;
  isDecisionBusy(task: Task, index: number): boolean;
}

// 时段试听的当前状态与触发入口（播放头绝对秒供切分面板绘制）。
export interface PlaybackApi {
  playing: Playing | null;
  playPos: number | null;
  playSegment(
    taskId: string,
    speaker: string,
    seg: Pick<TimelineSegment, "start" | "end">,
  ): void;
}
