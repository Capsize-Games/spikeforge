import { useCallback, useRef } from "react";

import { AnalysisPanels } from "./components/AnalysisPanels";
import { Controls } from "./components/Controls";
import { DownloadProgress } from "./components/DownloadProgress";
import { EnergyPanel } from "./components/EnergyPanel";
import { HubPanel } from "./components/HubPanel";
import { ModelPanel } from "./components/ModelPanel";
import { StatusBar } from "./components/StatusBar";
import { TopBar } from "./components/TopBar";
import { TourCard } from "./components/TourCard";
import { TrainingPanel } from "./components/TrainingPanel";
import { ViewerPanels } from "./components/ViewerPanels";
import { useEncodeConfig } from "./hooks/useEncodeConfig";
import { useEnergy } from "./hooks/useEnergy";
import { useHub } from "./hooks/useHub";
import { useModelActions } from "./hooks/useModelActions";
import { useServerBootstrap } from "./hooks/useServerBootstrap";
import { useTour } from "./hooks/useTour";
import { useViewer } from "./hooks/useViewer";
import { LESSONS } from "./tour/lessons";
import { useTraining } from "./useTraining";
import { useWebSocket } from "./useWebSocket";
import type { Modality, ServerMsg } from "./types";

export default function App() {
  const { config, configRef, patchConfig, replaceConfig } = useEncodeConfig();
  const training = useTraining();
  const tour = useTour(LESSONS);

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
  handlerRef.current = (msg) => {
    viewer.onMessage(msg);
    hub.onMessage(msg);
    energy.onMessage(msg);
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

  // The configured surrogate drives the curve panel's initial selection.
  const rawSurrogate = training.state.config.topology_params.surrogate;
  const currentSurrogate = typeof rawSurrogate === "string" ? rawSurrogate : "";

  const listed = training.state.datasets.find((d) => d.name === config.dataset);
  const modality: Modality = listed?.modality ?? "image";

  return (
    <div className="app">
      <header className="app-header">
        <TopBar
          mode={training.state.config.mode}
          onModeChange={(mode) => training.patch({ mode })}
          lessons={LESSONS}
          tourOpen={tour.menuOpen}
          onToggleTours={tour.toggleMenu}
          onOpenTour={tour.openLesson}
        />
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
              topologies={training.state.topologies}
              neurons={training.state.neurons}
              surrogates={training.state.surrogates}
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
            eventFrame={viewer.state.eventFrame}
            spikeFrame={viewer.displayFrame}
            reconGain1={viewer.state.reconGain1}
            rasters={viewer.state.rasters}
            inference={viewer.state.inference}
            modality={modality}
            hiddenFrame={viewer.state.spikeFrames.hidden}
            animation={viewer.state.animation}
            loaded={training.state.loaded}
            coding={config.coding}
            mode={training.state.config.mode}
            sampleIndex={config.sample_index}
            timeStep={viewer.timeStep}
            numSteps={viewer.numSteps}
            trajectory={viewer.state.trajectory}
            nirGraph={viewer.state.nirGraph}
            nirValidation={viewer.state.nirValidation}
            targetList={viewer.state.targetList}
            deploymentReport={viewer.state.deploymentReport}
            backendRun={viewer.state.backendRun}
            playing={viewer.state.running}
            onPlay={actions.playPreview}
            onStop={actions.stopPreview}
            onScrub={(step) => viewer.setScrubStep(step)}
            onSelectSample={(index) =>
              actions.selectSample({ sample_index: index })
            }
            onRefreshTrajectory={actions.requestTrajectory}
            onRefreshNirGraph={actions.requestNirExport}
            onRefreshNirValidation={actions.requestNirValidate}
            onRefreshTargets={actions.requestTargets}
            onSelectTarget={actions.requestDeploymentReport}
            onRunBackend={actions.requestDeployRun}
            onSwitchToEducational={() =>
              training.patch({ mode: "educational" })
            }
          />

          <div className="col-training">
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

            <AnalysisPanels
              mode={training.state.config.mode}
              metrics={viewer.state.metrics}
              encodingReport={viewer.state.encodingReport}
              surrogateCurve={viewer.state.surrogateCurve}
              benchmark={training.state.benchmark}
              benchmarkLoading={training.state.benchmarkLoading}
              surrogates={training.state.surrogates}
              currentSurrogate={currentSurrogate}
              onRefreshMetrics={actions.requestMetrics}
              onRefreshEncodingReport={actions.requestEncodingReport}
              onRequestSurrogateCurve={actions.requestSurrogateCurve}
              onRunBenchmark={actions.requestBenchmark}
              onSwitchMode={() => training.patch({ mode: "educational" })}
            />

            <HubPanel hub={hub} />

            <EnergyPanel
              payload={energy.payload}
              targets={viewer.state.targetList?.targets ?? []}
              loading={energy.loading}
              onRun={(target) =>
                energy.run(training.state.config, target)
              }
            />
          </div>
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
