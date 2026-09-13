import { lazy, Suspense, useCallback, useRef } from "react";

import { AnalysisPanels } from "./components/AnalysisPanels";
import { Controls } from "./components/Controls";
import { DownloadProgress } from "./components/DownloadProgress";
import { EnergyPanel } from "./components/EnergyPanel";
import { HubPanel } from "./components/HubPanel";
import { LoadedModelPanel } from "./components/LoadedModelPanel";
import { ModelPanel } from "./components/ModelPanel";
import { Section } from "./components/Stepper";
import { StatusBar } from "./components/StatusBar";
import { TabBar } from "./components/TabBar";
import { TabPanel } from "./components/TabPanel";
import { TargetsPanel } from "./components/TargetsPanel";
import { TopBar } from "./components/TopBar";
import { TourCard } from "./components/TourCard";
import { TrainControls } from "./components/TrainControls";
import { TrainingPanel } from "./components/TrainingPanel";
import { ViewerPanels } from "./components/ViewerPanels";
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
import type { Modality, ServerMsg } from "./types";

// React Flow is a real bundle addition (~100kb), so the Pipeline tab is the
// app's first code-split boundary -- everyone else's initial load stays
// exactly as it was.
const PipelinePanel = lazy(() =>
  import("./components/PipelinePanel").then((m) => ({
    default: m.PipelinePanel,
  })),
);

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

  // The configured surrogate drives the curve panel's initial selection.
  const rawSurrogate = training.state.config.topology_params.surrogate;
  const currentSurrogate = typeof rawSurrogate === "string" ? rawSurrogate : "";

  const listed = training.state.datasets.find((d) => d.name === config.dataset);
  const modality: Modality = listed?.modality ?? "image";
  const readOnly = viewer.state.status?.read_only === true;

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

        <TabBar
          tabs={TABS}
          active={tabs.active}
          busy={busy}
          onSelect={tabs.select}
        />

        {training.state.loaded && (
          <div className="loaded-model-bar">
            <LoadedModelPanel loaded={training.state.loaded} />
          </div>
        )}
      </header>

      <main className="app-main">
        <TabPanel id="model" active={tabs.active}>
          <div className="tab-cols tab-cols-model">
            <div className="tab-col">
              <ModelPanel
                models={training.state.models}
                current={training.state.loaded?.name ?? null}
                connected={ws.connected}
                busy={training.state.running}
                loading={viewer.modelLoading}
                readOnly={readOnly}
                onNew={actions.newModel}
                onLoad={actions.loadModel}
                onSave={actions.saveModel}
              />

              <div className="panel">
                <Section
                  title="Train & inspect"
                  hint="fit the network, manage checkpoints"
                >
                  <TrainControls
                    running={training.state.running}
                    connected={ws.connected}
                    readOnly={readOnly}
                    onTrain={actions.train}
                    onStop={actions.stopTrain}
                  />
                </Section>
              </div>
            </div>

            <div className="tab-col">
              <Controls
                config={config}
                model={training.state.config}
                datasets={training.state.datasets}
                gpuAvailable={viewer.gpuAvailable}
                topologies={training.state.topologies}
                neurons={training.state.neurons}
                surrogates={training.state.surrogates}
                locked={viewer.locked}
                onChange={patchConfig}
                onModelChange={training.patch}
                onSelectSample={actions.selectSample}
              />
            </div>
          </div>
        </TabPanel>

        <TabPanel id="viewer" active={tabs.active}>
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
            autoPredict={viewer.autoPredict}
            canAutoPredict={viewer.canAutoPredict}
            onToggleAutoPredict={toggleAutoPredict}
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
            onSwitchToEducational={() =>
              training.patch({ mode: "educational" })
            }
          />
        </TabPanel>

        <TabPanel id="training" active={tabs.active}>
          <div className="tab-cols">
            <div className="tab-col">
              <TrainingPanel
                loss={training.state.loss}
                trainAccuracy={training.state.trainAccuracy}
                testAccuracy={training.state.testAccuracy}
                last={training.state.last}
                device={training.state.device}
              />
            </div>

            <div className="tab-col">
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
            </div>
          </div>
        </TabPanel>

        <TabPanel id="hub" active={tabs.active}>
          <div className="tab-col narrow">
            <HubPanel hub={hub} />
          </div>
        </TabPanel>

        <TabPanel id="deploy" active={tabs.active}>
          <div className="tab-cols">
            <div className="tab-col">
              <TargetsPanel
                list={viewer.state.targetList}
                report={viewer.state.deploymentReport}
                backendRun={viewer.state.backendRun}
                onRefresh={actions.requestTargets}
                onSelectTarget={actions.requestDeploymentReport}
                onRunBackend={actions.requestDeployRun}
              />
            </div>

            <div className="tab-col">
              <EnergyPanel
                payload={energy.payload}
                targets={viewer.state.targetList?.targets ?? []}
                loading={energy.loading}
                onRun={(target) => energy.run(training.state.config, target)}
              />
            </div>
          </div>
        </TabPanel>

        <TabPanel id="pipeline" active={tabs.active}>
          <Suspense fallback={<div className="muted">Loading pipeline editor…</div>}>
            <PipelinePanel
              models={training.state.models}
              connected={ws.connected}
              graph={pipeline.state.graph}
              pipelines={pipeline.state.pipelines}
              running={pipeline.state.running}
              nodeStatus={pipeline.state.nodeStatus}
              nodeResult={pipeline.state.nodeResult}
              status={pipeline.state.status}
              onListPipelines={() => ws.sendAction("list_pipelines")}
              onSavePipeline={(name) =>
                ws.sendSavePipeline(name, pipeline.state.graph)
              }
              onLoadPipeline={(name) => ws.sendPipelineName("load_pipeline", name)}
              onDeletePipeline={(name) =>
                ws.sendPipelineName("delete_pipeline", name)
              }
              onRunPipeline={(input) =>
                ws.sendRunPipeline(pipeline.state.graph, input)
              }
              onStopPipeline={() => ws.sendAction("stop_pipeline")}
              onSetGraph={pipeline.setGraph}
              onAddNode={pipeline.addNode}
              onMoveNode={pipeline.moveNode}
              onRemoveNode={pipeline.removeNode}
              onAddEdge={pipeline.addEdge}
              onRemoveEdge={pipeline.removeEdge}
              onSetEdgeExtract={pipeline.setEdgeExtract}
            />
          </Suspense>
        </TabPanel>

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
