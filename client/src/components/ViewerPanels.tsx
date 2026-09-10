import type { InferencePayload, RasterPayload, RasterSource } from "../types";
import { HeatmapCanvas } from "./HeatmapCanvas";
import { RasterCanvas } from "./RasterCanvas";

interface Props {
  sample: number[][] | null;
  spikeFrame: number[][] | null;
  reconGain1: number[][] | null;
  reconLow: number[][] | null;
  rasters: Record<RasterSource, RasterPayload | null>;
  inference: InferencePayload | null;
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

export function ViewerPanels({
  sample,
  spikeFrame,
  reconGain1,
  reconLow,
  rasters,
  inference,
}: Props) {
  return (
    <div className="col-viz">
      <div className="row viz-row">
        <div className="panel grow">
          <div className="panel-title">Input</div>
          <div className="pair grow">
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
          </div>
        </div>

        <div className="panel grow">
          <div className="panel-title">Reconstruction</div>
          <div className="pair grow">
            <HeatmapCanvas
              data={reconGain1}
              palette="binary"
              label="Gain=1"
              width={120}
              height={120}
              fluid
            />
            <HeatmapCanvas
              data={reconLow}
              palette="binary"
              label="Low gain"
              width={120}
              height={120}
              fluid
            />
          </div>
        </div>
      </div>

      <RasterCanvas raster={rasters.input} label="Input spikes" />
      <RasterCanvas raster={rasters.hidden} label="Hidden layer" />
      <RasterCanvas raster={rasters.output} label="Output layer" />
    </div>
  );
}
