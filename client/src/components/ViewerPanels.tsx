import type { TrajectoryPayload } from "../introspectionTypes";
import type { NirGraphPayload, NirValidationPayload } from "../nirTypes";
import type {
  DeploymentReportPayload,
  TargetListPayload,
} from "../targetTypes";
import type {
  CodingType,
  ExecutionMode,
  InferencePayload,
  Modality,
  ModelLoadedPayload,
  RasterPayload,
  RasterSource,
} from "../types";
import { InputRow } from "./InputRow";
import { LoadedModelPanel } from "./LoadedModelPanel";
import { NetworkActivity } from "./NetworkActivity";
import { NirGraphPanel } from "./NirGraphPanel";
import { NirValidationPanel } from "./NirValidationPanel";
import { TargetsPanel } from "./TargetsPanel";
import { TimeCursor } from "./TimeCursor";
import { TrajectoryPanel } from "./TrajectoryPanel";

interface Props {
  sample: number[][] | null;
  eventFrame: number[][] | null;
  spikeFrame: number[][] | null;
  reconGain1: number[][] | null;
  rasters: Record<RasterSource, RasterPayload | null>;
  modality: Modality;
  inference: InferencePayload | null;
  loaded: ModelLoadedPayload | null;
  coding: CodingType;
  mode: ExecutionMode;
  sampleIndex: number;
  timeStep: number | null;
  numSteps: number;
  trajectory: TrajectoryPayload | null;
  nirGraph: NirGraphPayload | null;
  nirValidation: NirValidationPayload | null;
  targetList: TargetListPayload | null;
  deploymentReport: DeploymentReportPayload | null;
  playing: boolean;
  onPlay: () => void;
  onStop: () => void;
  onScrub: (step: number) => void;
  onSelectSample: (index: number) => void;
  onRefreshTrajectory: () => void;
  onRefreshNirGraph: () => void;
  onRefreshNirValidation: () => void;
  onRefreshTargets: () => void;
  onSelectTarget: (name: string) => void;
  onSwitchToEducational: () => void;
}

/** The middle column: model summary, cursor, input, activity, NIR panels. */
export function ViewerPanels({
  sample,
  eventFrame,
  spikeFrame,
  reconGain1,
  rasters,
  modality,
  inference,
  loaded,
  coding,
  mode,
  sampleIndex,
  timeStep,
  numSteps,
  trajectory,
  nirGraph,
  nirValidation,
  targetList,
  deploymentReport,
  playing,
  onPlay,
  onStop,
  onScrub,
  onSelectSample,
  onRefreshTrajectory,
  onRefreshNirGraph,
  onRefreshNirValidation,
  onRefreshTargets,
  onSelectTarget,
  onSwitchToEducational,
}: Props) {
  return (
    <div className="col-viz">
      {loaded && <LoadedModelPanel loaded={loaded} />}

      <TimeCursor
        step={timeStep}
        numSteps={numSteps}
        playing={playing}
        sampleIndex={sampleIndex}
        onScrub={onScrub}
        onPlay={onPlay}
        onStop={onStop}
        onSelectSample={onSelectSample}
      />

      <InputRow
        sample={sample}
        eventFrame={eventFrame}
        spikeFrame={spikeFrame}
        reconGain1={reconGain1}
        inference={inference}
        coding={coding}
        modality={modality}
      />

      <NetworkActivity
        rasters={rasters}
        inference={inference}
        timeStep={timeStep}
        modality={modality}
      />

      <TrajectoryPanel
        mode={mode}
        trajectory={trajectory}
        cursorIndex={timeStep}
        onRefresh={onRefreshTrajectory}
        onSwitchMode={onSwitchToEducational}
      />

      <NirGraphPanel graph={nirGraph} onRefresh={onRefreshNirGraph} />

      <NirValidationPanel
        validation={nirValidation}
        onRefresh={onRefreshNirValidation}
      />

      <TargetsPanel
        list={targetList}
        report={deploymentReport}
        onRefresh={onRefreshTargets}
        onSelectTarget={onSelectTarget}
      />
    </div>
  );
}
