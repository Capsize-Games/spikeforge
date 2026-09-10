import type {
  CodingType,
  InferencePayload,
  ModelLoadedPayload,
  RasterPayload,
  RasterSource,
} from "../types";
import { InputRow } from "./InputRow";
import { LoadedModelPanel } from "./LoadedModelPanel";
import { NetworkActivity } from "./NetworkActivity";
import { TimeCursor } from "./TimeCursor";

interface Props {
  sample: number[][] | null;
  spikeFrame: number[][] | null;
  reconGain1: number[][] | null;
  rasters: Record<RasterSource, RasterPayload | null>;
  inference: InferencePayload | null;
  loaded: ModelLoadedPayload | null;
  coding: CodingType;
  sampleIndex: number;
  timeStep: number | null;
  numSteps: number;
  playing: boolean;
  onPlay: () => void;
  onStop: () => void;
  onScrub: (step: number) => void;
  onSelectSample: (index: number) => void;
}

/** The middle column: loaded model summary, time cursor, input row, rasters. */
export function ViewerPanels({
  sample,
  spikeFrame,
  reconGain1,
  rasters,
  inference,
  loaded,
  coding,
  sampleIndex,
  timeStep,
  numSteps,
  playing,
  onPlay,
  onStop,
  onScrub,
  onSelectSample,
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
        spikeFrame={spikeFrame}
        reconGain1={reconGain1}
        inference={inference}
        coding={coding}
      />

      <NetworkActivity
        rasters={rasters}
        inference={inference}
        timeStep={timeStep}
      />
    </div>
  );
}
