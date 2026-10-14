import { AnalysisPanels } from "./AnalysisPanels";
import { ModelWorkspace } from "./ModelWorkspace";
import { TabPanel } from "./TabPanel";
import { TrainingPanel } from "./TrainingPanel";
import { ViewerPanels } from "./ViewerPanels";
import { useModelActions } from "../hooks/useModelActions";
import { useViewer } from "../hooks/useViewer";
import type { TabId } from "../tabs";
import type { EncodeConfig, Modality } from "../types";
import { useTraining } from "../useTraining";

type Training = ReturnType<typeof useTraining>;
type Viewer = ReturnType<typeof useViewer>;
type Actions = ReturnType<typeof useModelActions>;

interface Props {
  active: TabId;
  config: EncodeConfig;
  training: Training;
  viewer: Viewer;
  actions: Actions;
  connected: boolean;
  onPatchConfig: (patch: Partial<EncodeConfig>) => void;
  onToggleAutoPredict: () => void;
}

/** The model, viewer, and training tabs of the dashboard. */
export function WorkspaceTabs({
  active,
  config,
  training,
  viewer,
  actions,
  connected,
  onPatchConfig,
  onToggleAutoPredict,
}: Props) {
  // The configured surrogate drives the curve panel's initial selection.
  const rawSurrogate = training.state.config.topology_params.surrogate;
  const currentSurrogate = typeof rawSurrogate === "string" ? rawSurrogate : "";

  const listed = training.state.datasets.find((d) => d.name === config.dataset);
  const modality: Modality = listed?.modality ?? "image";
  const readOnly = viewer.state.status?.read_only === true;

  return (
    <>
      <TabPanel id="model" active={active}>
        <ModelWorkspace
          config={config}
          model={training.state.config}
          datasets={training.state.datasets}
          models={training.state.models}
          currentModel={training.state.loaded?.name ?? null}
          loaded={training.state.loaded}
          sample={viewer.state.sample}
          eventFrame={viewer.state.eventFrame}
          topologies={training.state.topologies}
          neurons={training.state.neurons}
          surrogates={training.state.surrogates}
          gpuAvailable={viewer.gpuAvailable}
          connected={connected}
          busy={training.state.running}
          loading={viewer.modelLoading}
          readOnly={readOnly}
          locked={viewer.locked}
          onPatchConfig={onPatchConfig}
          onModelChange={training.patch}
          onSelectSample={actions.selectSample}
          onNew={actions.newModel}
          onLoad={actions.loadModel}
          onSave={actions.saveModel}
          onTrain={actions.train}
          onStopTrain={actions.stopTrain}
        />
      </TabPanel>

      <TabPanel id="viewer" active={active}>
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
          onToggleAutoPredict={onToggleAutoPredict}
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

      <TabPanel id="training" active={active}>
        <div className="tab-cols tab-cols-training">
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
    </>
  );
}
