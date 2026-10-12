// 应用外壳：组装各 hook 的状态与副作用，把页面交给三个区块组件渲染。
import { SubmitForm } from "./components/SubmitForm";
import { TaskList } from "./components/TaskList";
import { WorkspacePanel } from "./components/WorkspacePanel";
import { useCorrections } from "./hooks/useCorrections";
import { useSegmentPlayback } from "./hooks/useSegmentPlayback";
import { useServerPolling } from "./hooks/useServerPolling";
import { useSpeakerNaming } from "./hooks/useSpeakerNaming";
import { useSubmitForm } from "./hooks/useSubmitForm";
import { useTaskActions } from "./hooks/useTaskActions";
import { useTimelineStore } from "./hooks/useTimelineStore";
import { useVoiceprintCheck } from "./hooks/useVoiceprintCheck";
import { useVoiceprintLibrary } from "./hooks/useVoiceprintLibrary";
import { useWorkspaces } from "./hooks/useWorkspaces";
import type { CheckActions, NamingActions, PlaybackApi } from "./types";

export default function App() {
  const timelineStore = useTimelineStore();
  const voiceprintCheck = useVoiceprintCheck({
    applyTimeline: timelineStore.applyTimeline,
  });
  const polling = useServerPolling({
    loadTimeline: timelineStore.loadTimeline,
    syncCheckState: voiceprintCheck.maybeFetchCheck,
  });
  const corrections = useCorrections({
    applyTimeline: timelineStore.applyTimeline,
  });
  const naming = useSpeakerNaming({
    timelines: timelineStore.timelines,
    speakerNames: timelineStore.speakerNames,
    applySpeakerName: timelineStore.applySpeakerName,
    refreshVoiceprints: polling.refreshVoiceprints,
  });
  const library = useVoiceprintLibrary({
    refreshVoiceprints: polling.refreshVoiceprints,
  });
  const workspaceActions = useWorkspaces({ refresh: polling.refresh });
  const taskActions = useTaskActions({ refresh: polling.refresh });
  const submitForm = useSubmitForm({ refresh: polling.refresh });
  const playback = useSegmentPlayback();

  const namingActions: NamingActions = {
    renameSpeaker: naming.renameSpeaker,
    storeVoiceprint: naming.handleStoreVoiceprint,
    isVoiceprintBusy: (task, label) =>
      Boolean(naming.voiceprintBusy[`${task.id}:${label}`]),
  };
  const checkActions: CheckActions = {
    startCheck: voiceprintCheck.startVoiceprintCheck,
    cancelCheck: voiceprintCheck.cancelVoiceprintCheck,
    decideSuggestion: voiceprintCheck.decideSuggestion,
    isDecisionBusy: (task, index) =>
      Boolean(voiceprintCheck.decisionBusy[`${task.id}:${index}`]),
  };
  const playbackApi: PlaybackApi = {
    playing: playback.playing,
    playPos: playback.playPos,
    playSegment: playback.playSegment,
  };

  return (
    <main className="app">
      <h1>视频人声擦除</h1>

      <SubmitForm {...submitForm} />

      <TaskList
        tasks={polling.tasks}
        resumeError={taskActions.resumeError}
        checkError={voiceprintCheck.checkError}
        timelines={timelineStore.timelines}
        speakerNames={timelineStore.speakerNames}
        transcripts={polling.transcripts}
        ingestWarnings={naming.ingestWarnings}
        correctionError={corrections.correctionError}
        checkStates={voiceprintCheck.checkStates}
        onCancel={taskActions.handleCancel}
        onResume={taskActions.handleResume}
        corrections={corrections}
        naming={namingActions}
        checks={checkActions}
        playback={playbackApi}
      />

      <WorkspacePanel
        workspaces={polling.workspaces}
        workspaceError={workspaceActions.workspaceError}
        voiceprints={polling.voiceprints}
        voiceprintError={library.voiceprintError}
        tasks={polling.tasks}
        handleResumeWorkspace={workspaceActions.handleResumeWorkspace}
        handleOpenWorkspace={workspaceActions.handleOpenWorkspace}
        handleDeleteWorkspace={workspaceActions.handleDeleteWorkspace}
        onDeleteVoiceprint={library.handleDeleteVoiceprint}
      />
    </main>
  );
}
