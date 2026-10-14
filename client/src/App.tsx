import { useCallback, useRef } from "react";

import { AppHeader } from "./components/AppHeader";
import { DownloadProgress } from "./components/DownloadProgress";
import { NavRail } from "./components/NavRail";
import { OperationsTabs } from "./components/OperationsTabs";
import { StatusBar } from "./components/StatusBar";
import { TourCard } from "./components/TourCard";
import { WorkspaceTabs } from "./components/WorkspaceTabs";
import { useEncodeConfig } from "./hooks/useEncodeConfig";
import { useEnergy } from "./hooks/useEnergy";
import { useHub } from "./hooks/useHub";
import { useModelActions } from "./hooks/useModelActions";
import { useServerBootstrap } from "./hooks/useServerBootstrap";
import { useTabs } from "./hooks/useTabs";
import { useTour } from "./hooks/useTour";
import { useViewer } from "./hooks/useViewer";
import { usePipeline } from "./usePipeline";
import { TABS } from "./tabs";
import { LESSONS } from "./tour/lessons";
import { useTraining } from "./useTraining";
import { useWebSocket } from "./useWebSocket";
import type { TabId } from "./tabs";
import type { ServerMsg } from "./types";

export default function App() {
  const { config, configRef, patchConfig, replaceConfig } = useEncodeConfig();
  const training = useTraining();
  const tour = useTour(LESSONS);
  // The open tour step also drives the tab switch, so a step can point at any
  // panel without the lesson files knowing about tabs.
  const tabs = useTabs(tour.step?.target ?? null);

  // useViewer needs the socket's send helpers, while the socket needs
  // useViewer's message handler — bridge the cycle with a ref.
  const handlerRef = useRef<((msg: ServerMsg) => void) | null>(null);
  const onMessage = useCallback((msg: ServerMsg) => {
    handlerRef.current?.(msg);
  }, []);
  const ws = useWebSocket({ onMessage });

  const viewer = useViewer({
    config,
    configRef,
    training,
    connected: ws.connected,
    sendInfer: ws.sendInfer,
  });
  const hub = useHub({ ws, connected: ws.connected });
  const energy = useEnergy({ ws, connected: ws.connected });
  const pipeline = usePipeline();
  handlerRef.current = (msg) => {
    viewer.onMessage(msg);
    hub.onMessage(msg);
    energy.onMessage(msg);
    pipeline.handleMessage(msg);
  };

  const actions = useModelActions({
    config,
    configRef,
    replaceConfig,
    training,
    ws,
    setModelLoading: viewer.setModelLoading,
    setScrubStep: viewer.setScrubStep,
  });

  useServerBootstrap(ws, configRef);

  /** Toggle auto-prediction: run now when enabled, clear when disabled. */
  const toggleAutoPredict = () => {
    const next = !viewer.autoPredict;
    viewer.setAutoPredict(next);
    if (next) {
      if (viewer.canAutoPredict) actions.runInference();
    } else {
      viewer.clearInference();
    }
  };

  const downloading =
    viewer.state.download?.status === "downloading"
      ? viewer.state.download
      : null;

  // Background work keeps running on an unfocused tab; mark it in the strip.
  const busy: Partial<Record<TabId, boolean>> = {
    training: training.state.running,
    hub: downloading !== null,
    pipeline: pipeline.state.running,
  };

  return (
    <div className="app">
      <NavRail
        tabs={TABS}
        active={tabs.active}
        busy={busy}
        onSelect={tabs.select}
      />

      <AppHeader
        mode={training.state.config.mode}
        onModeChange={(mode) => training.patch({ mode })}
        active={tabs.active}
        loaded={training.state.loaded}
        dataset={config.dataset}
        tourOpen={tour.menuOpen}
        onToggleTours={tour.toggleMenu}
        onOpenTour={tour.openLesson}
      />

      <main className="app-main">
        <WorkspaceTabs
          active={tabs.active}
          config={config}
          training={training}
          viewer={viewer}
          actions={actions}
          connected={ws.connected}
          onPatchConfig={patchConfig}
          onToggleAutoPredict={toggleAutoPredict}
        />

        <OperationsTabs
          active={tabs.active}
          ws={ws}
          hub={hub}
          energy={energy}
          pipeline={pipeline}
          training={training}
          viewer={viewer}
          actions={actions}
        />

        {viewer.state.error && (
          <div className="error">{viewer.state.error}</div>
        )}
      </main>

      <StatusBar
        stats={viewer.stats}
        requested={training.state.config.device}
        connected={ws.connected}
        unauthorized={ws.unauthorized}
      />

      {downloading && (
        <DownloadProgress
          download={downloading}
          onCancel={ws.sendCancelDownload}
        />
      )}

      {tour.lesson && tour.step && (
        <TourCard
          lesson={tour.lesson}
          step={tour.step}
          stepIndex={tour.stepIndex}
          missing={tour.missing}
          onPrev={tour.prev}
          onNext={tour.next}
          onClose={tour.closeLesson}
        />
      )}
    </div>
  );
}
