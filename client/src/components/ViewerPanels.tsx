import type { TrajectoryPayload } from "../introspectionTypes";
import type { NirGraphPayload, NirValidationPayload } from "../nirTypes";
import type {
  AnimationStatePayload,
  CodingType,
  ExecutionMode,
  InferencePayload,
  Modality,
  ModelLoadedPayload,
  RasterPayload,
  RasterSource,
} from "../types";
import { useI18n } from "../i18n/I18nProvider";
import { InputRow } from "./InputRow";
import { InspectorSection } from "./InspectorSection";
import { NetworkActivity } from "./NetworkActivity";
import { NirGraphPanel } from "./NirGraphPanel";
import { NirValidationPanel } from "./NirValidationPanel";
import { PredictionPanel } from "./PredictionPanel";
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
  hiddenFrame: number[][] | null;
  animation: AnimationStatePayload | null;
  loaded: ModelLoadedPayload | null;
  coding: CodingType;
  mode: ExecutionMode;
  sampleIndex: number;
  timeStep: number | null;
  numSteps: number;
  trajectory: TrajectoryPayload | null;
  nirGraph: NirGraphPayload | null;
  nirValidation: NirValidationPayload | null;
  /** Whether auto-prediction is switched on. */
  autoPredict: boolean;
  /** Whether a prediction can run at all (model + connection). */
  canAutoPredict: boolean;
  onToggleAutoPredict: () => void;
  playing: boolean;
  onPlay: () => void;
  onStop: () => void;
  onScrub: (step: number) => void;
  onSelectSample: (index: number) => void;
  onRefreshTrajectory: () => void;
  onRefreshNirGraph: () => void;
  onRefreshNirValidation: () => void;
  onSwitchToEducational: () => void;
}

/**
 * The viewer: the app's hero workspace. The stage (input samples, then the
 * per-layer network activity) takes most of the width; the inspector column
 * holds the prediction and the on-demand analysis panels; the transport strip
 * runs the full width along the bottom.
 */
export function ViewerPanels({
  sample,
  eventFrame,
  spikeFrame,
  reconGain1,
  rasters,
  modality,
  inference,
  hiddenFrame,
  animation,
  loaded,
  coding,
  mode,
  sampleIndex,
  timeStep,
  numSteps,
  trajectory,
  nirGraph,
  nirValidation,
  autoPredict,
  canAutoPredict,
  onToggleAutoPredict,
  playing,
  onPlay,
  onStop,
  onScrub,
  onSelectSample,
  onRefreshTrajectory,
  onRefreshNirGraph,
  onRefreshNirValidation,
  onSwitchToEducational,
}: Props) {
  const { t } = useI18n();
  return (
    <div className="viewer-workspace">
      <div className="viewer-stage">
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
          hiddenFrame={hiddenFrame}
          animation={animation}
        />
      </div>

      <InspectorSection
        title={t("viewer.inspector")}
        className="viewer-inspector"
      >
        <PredictionPanel
          inference={inference}
          timeStep={timeStep}
          loaded={loaded}
          autoPredict={autoPredict}
          canAutoPredict={canAutoPredict}
          onToggleAutoPredict={onToggleAutoPredict}
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
      </InspectorSection>

      <div className="viewer-transport">
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
        {/* What the execution mode does to capture, stated where the captured
            frames are rather than in the global toolbar. */}
        <span className="capture-note" data-testid="capture-state">
          {mode === "educational" ? t("mode.captureOn") : t("mode.captureOff")}
        </span>
      </div>
    </div>
  );
}
