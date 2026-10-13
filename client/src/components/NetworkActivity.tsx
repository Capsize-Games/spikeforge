import { HELP } from "../helpText";
import type {
  AnimationStatePayload,
  InferencePayload,
  Modality,
  RasterPayload,
  RasterSource,
} from "../types";
import { HiddenFramePanel } from "./HiddenFramePanel";
import { PanelHeader } from "./PanelHeader";
import { RasterCanvas } from "./RasterCanvas";

interface Props {
  rasters: Record<RasterSource, RasterPayload | null>;
  inference: InferencePayload | null;
  timeStep: number | null;
  modality: Modality;
  /** Latest streamed hidden-layer frame, when animation is enabled. */
  hiddenFrame: number[][] | null;
  /** Availability of the hidden animation stream, once requested. */
  animation: AnimationStatePayload | null;
}

/** Compact per-layer aggregate for training diagnostics. */
function summarize(raster: RasterPayload | null): string | undefined {
  if (!raster) return undefined;
  const total = raster.time.length;
  const unique = new Set(raster.neurons).size;
  const perStep = raster.num_steps
    ? (total / raster.num_steps).toFixed(1)
    : "0";
  const pct = raster.num_neurons
    ? Math.round((unique / raster.num_neurons) * 100)
    : 0;
  const silent = Math.max(0, raster.num_neurons - unique);
  return (
    `${perStep} spikes/step · ${unique}/${raster.num_neurons} active` +
    ` (${pct}%) · ${silent} silent`
  );
}

/**
 * The input/hidden/output rasters stacked under one shared panel header.
 * Merging them removes the duplicated per-panel titles and padding that made
 * the middle column the tallest, while keeping every layer visible at once.
 */
export function NetworkActivity({
  rasters,
  inference,
  timeStep,
  modality,
  hiddenFrame,
  animation,
}: Props) {
  const eventInput = modality === "event";
  const inputLabel = eventInput
    ? "Event spikes · neuron (ON then OFF)"
    : "Input spikes · neuron";
  const outputLabels = rasters.output
    ? Array.from({ length: rasters.output.num_neurons }, (_, i) => String(i))
    : undefined;
  const outputRows = inference
    ? [inference.predicted, inference.true_label].filter(
        (v): v is number => v !== null && v >= 0,
      )
    : undefined;

  return (
    <div className="panel activity-panel">
      <PanelHeader title="Network activity" hint={HELP.activity} />
      <div className="raster-stack">
        <RasterCanvas
          bare
          height={185}
          raster={rasters.input}
          label={inputLabel}
          summary={summarize(rasters.input)}
          highlightStep={timeStep}
          emptyNote={eventInput ? "select an event sample" : undefined}
        />
        {(animation || hiddenFrame) && (
          <HiddenFramePanel
            frame={hiddenFrame}
            step={timeStep}
            unavailable={
              animation && !animation.available ? animation.reason : null
            }
          />
        )}
        <RasterCanvas
          bare
          height={130}
          raster={rasters.hidden}
          label="Hidden layer · neuron (sorted by rate)"
          summary={summarize(rasters.hidden)}
          sortByRate
          highlightStep={timeStep}
          emptyNote="train or load a model to see layer activity"
        />
        <RasterCanvas
          bare
          height={130}
          raster={rasters.output}
          label="Output layer · class"
          yLabels={outputLabels}
          highlightRows={outputRows}
          summary={summarize(rasters.output)}
          highlightStep={timeStep}
          emptyNote="train or load a model to see layer activity"
        />
      </div>
    </div>
  );
}
