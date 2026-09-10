import { useCallback, useEffect, useRef } from "react";

import { Controls } from "./components/Controls";
import { DownloadProgress } from "./components/DownloadProgress";
import { ModelPanel } from "./components/ModelPanel";
import { StatusBar } from "./components/StatusBar";
import { TopBar } from "./components/TopBar";
import { TrainingPanel } from "./components/TrainingPanel";
import { ViewerPanels } from "./components/ViewerPanels";
import { useEncodeConfig } from "./hooks/useEncodeConfig";
import { useModelActions } from "./hooks/useModelActions";
import { useViewer } from "./hooks/useViewer";
import { useTraining } from "./useTraining";
import { useWebSocket } from "./useWebSocket";
import type { ServerMsg } from "./types";

export default function App() {
  const { config, configRef, patchConfig, replaceConfig } = useEncodeConfig();
  const training = useTraining();

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
  handlerRef.current = viewer.onMessage;

  const actions = useModelActions({
    config,
    configRef,
    replaceConfig,
    training,
    ws,
    setModelLoading: viewer.setModelLoading,
    setScrubStep: viewer.setScrubStep,
  });

  useEffect(() => {
    if (!ws.connected) return;
    ws.sendNamed("list_models", "");
    // Populate the viewer immediately so every panel is shown at rest.
    ws.send("configure", configRef.current);
  }, [ws.connected, ws.sendNamed, ws.send, configRef]);

  useEffect(() => {
    if (!ws.connected) return;
    ws.sendStats();
    const id = setInterval(ws.sendStats, 2000);
    return () => clearInterval(id);
  }, [ws.connected, ws.sendStats]);

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

  return (
    <div className="app">
      <header className="app-header">
        <TopBar />
      </header>

      <main className="app-main">
        <div className="grid">
          <div className="col-controls">
            <ModelPanel
              models={training.state.models}
              current={training.state.loaded?.name ?? null}
              connected={ws.connected}
              busy={training.state.running}
              loading={viewer.modelLoading}
              onNew={actions.newModel}
              onLoad={actions.loadModel}
              onSave={actions.saveModel}
            />
            <Controls
              config={config}
              model={training.state.config}
              datasets={training.state.datasets}
              gpuAvailable={viewer.gpuAvailable}
              connected={ws.connected}
              trainRunning={training.state.running}
              locked={viewer.locked}
              onChange={patchConfig}
              onModelChange={training.patch}
              onSelectSample={actions.selectSample}
              onTrain={actions.train}
              onStopTrain={actions.stopTrain}
            />
          </div>

          <ViewerPanels
            sample={viewer.state.sample}
            spikeFrame={viewer.displayFrame}
            reconGain1={viewer.state.reconGain1}
            rasters={viewer.state.rasters}
            inference={viewer.state.inference}
            loaded={training.state.loaded}
            coding={config.coding}
            sampleIndex={config.sample_index}
            timeStep={viewer.timeStep}
            numSteps={viewer.numSteps}
            playing={viewer.state.running}
            onPlay={actions.playPreview}
            onStop={actions.stopPreview}
            onScrub={(step) => viewer.setScrubStep(step)}
            onSelectSample={(index) =>
              actions.selectSample({ sample_index: index })
            }
          />

          <TrainingPanel
            loss={training.state.loss}
            trainAccuracy={training.state.trainAccuracy}
            testAccuracy={training.state.testAccuracy}
            last={training.state.last}
            device={training.state.device}
            inference={viewer.state.inference}
            timeStep={viewer.timeStep}
            loaded={training.state.loaded}
            autoPredict={viewer.autoPredict}
            canAutoPredict={viewer.canAutoPredict}
            onToggleAutoPredict={toggleAutoPredict}
          />
        </div>

        {viewer.state.error && (
          <div className="error">{viewer.state.error}</div>
        )}
      </main>

      <StatusBar
        stats={viewer.stats}
        requested={training.state.config.device}
        connected={ws.connected}
      />

      {downloading && (
        <DownloadProgress
          download={downloading}
          onCancel={ws.sendCancelDownload}
        />
      )}
    </div>
  );
}
