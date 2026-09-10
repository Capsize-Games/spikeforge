import type {
  CodingType,
  InferencePayload,
  RasterPayload,
  RasterSource,
} from "../types";
import { HeatmapCanvas } from "./HeatmapCanvas";
import { RasterCanvas } from "./RasterCanvas";
import { TimeCursor } from "./TimeCursor";

interface Props {
  sample: number[][] | null;
  spikeFrame: number[][] | null;
  reconGain1: number[][] | null;
  rasters: Record<RasterSource, RasterPayload | null>;
  inference: InferencePayload | null;
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

/** Predicted vs true badge shown while a model has scored the sample. */
function PredictionBadge({ inference }: { inference: InferencePayload }) {
  const trueLabel = inference.true_label;
  const correct = trueLabel === null || trueLabel === inference.predicted;
  return (
    <div className={`pred-badge ${correct ? "ok" : "bad"}`}>
      <span className="pred-badge-main">pred {inference.predicted}</span>
      <span className="pred-badge-conf">
        {(inference.confidence * 100).toFixed(1)}%
      </span>
      {trueLabel !== null && (
        <span className="pred-badge-true">true {trueLabel}</span>
      )}
    </div>
  );
}

/** Compact per-layer aggregate for training diagnostics. */
function summarize(raster: RasterPayload | null): string | undefined {
  if (!raster) return undefined;
  const total = raster.time.length;
  const unique = new Set(raster.neurons).size;
  const perStep = raster.num_steps ? (total / raster.num_steps).toFixed(1) : "0";
  const pct = raster.num_neurons
    ? Math.round((unique / raster.num_neurons) * 100)
    : 0;
  const silent = Math.max(0, raster.num_neurons - unique);
  return `${perStep} spikes/step · ${unique}/${raster.num_neurons} active (${pct}%) · ${silent} silent`;
}

export function ViewerPanels({
  sample,
  spikeFrame,
  reconGain1,
  rasters,
  inference,
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
  const rateCoding = coding === "rate";
  const outputLabels = rasters.output
    ? Array.from({ length: rasters.output.num_neurons }, (_, i) => String(i))
    : undefined;
  const outputRows = inference
    ? [inference.predicted, inference.true_label].filter(
        (v): v is number => v !== null && v >= 0,
      )
    : undefined;

  return (
    <div className="col-viz">
      <div className="viz-row">
        <div className="panel grow">
          <div className="panel-title">Input</div>
          <div className={`pair grow${rateCoding ? " three" : ""}`}>
            <div className="sample-wrap grow">
              <HeatmapCanvas
                data={sample}
                palette="binary"
                label="Sample"
                width={224}
                height={224}
                fluid
              />
              {inference && sample && <PredictionBadge inference={inference} />}
            </div>
            <HeatmapCanvas
              data={spikeFrame}
              palette="plasma"
              label="Spike frame"
              width={224}
              height={224}
              fluid
            />
            {rateCoding && (
              <HeatmapCanvas
                data={reconGain1}
                palette="binary"
                label="Decoded"
                width={224}
                height={224}
                fluid
              />
            )}
          </div>
        </div>
      </div>

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

      <RasterCanvas
        raster={rasters.input}
        label="Input spikes · neuron"
        summary={summarize(rasters.input)}
        highlightStep={timeStep}
      />

      <RasterCanvas
        raster={rasters.hidden}
        label="Hidden layer · neuron (sorted by rate)"
        summary={summarize(rasters.hidden)}
        sortByRate
        highlightStep={timeStep}
        emptyNote="train or load a model to see layer activity"
      />
      <RasterCanvas
        raster={rasters.output}
        label="Output layer · class"
        yLabels={outputLabels}
        highlightRows={outputRows}
        summary={summarize(rasters.output)}
        highlightStep={timeStep}
        emptyNote="train or load a model to see layer activity"
      />
    </div>
  );
}
