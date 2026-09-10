import type {
  CodingType,
  InferencePayload,
  Modality,
} from "../types";
import { HeatmapCanvas } from "./HeatmapCanvas";

interface Props {
  sample: number[][] | null;
  /** Whole-sample ON/OFF frame for event datasets. */
  eventFrame: number[][] | null;
  spikeFrame: number[][] | null;
  reconGain1: number[][] | null;
  inference: InferencePayload | null;
  coding: CodingType;
  modality: Modality;
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

/** Event modality: the ON/OFF frame plus the playing playback frame. */
function EventRow({ eventFrame, spikeFrame, inference }: Props) {
  return (
    <div className="viz-row" data-tour="input-raster">
      <div className="panel grow">
        <div className="pair grow">
          <div className="sample-wrap grow">
            <HeatmapCanvas
              data={eventFrame}
              palette="plasma"
              label="Event frame · ON top / OFF bottom"
              width={224}
              height={224}
              fluid
            />
            {inference && eventFrame && (
              <PredictionBadge inference={inference} />
            )}
          </div>
          <HeatmapCanvas
            data={spikeFrame}
            palette="plasma"
            label="Playback frame"
            width={224}
            height={224}
            fluid
          />
        </div>
      </div>
    </div>
  );
}

/** Image modality: raw sample, its spike frame, and any rate decode. */
function ImageRow({
  sample,
  spikeFrame,
  reconGain1,
  inference,
  coding,
}: Props) {
  const rateCoding = coding === "rate";
  return (
    <div className="viz-row" data-tour="input-raster">
      <div className="panel grow">
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
  );
}

/**
 * The input panel. Image datasets show the raw sample, its spike frame,
 * and (for rate coding) the decoded reconstruction; event datasets show
 * the recording's own ON/OFF frame and playback instead.
 */
export function InputRow(props: Props) {
  if (props.modality === "event") return <EventRow {...props} />;
  return <ImageRow {...props} />;
}
