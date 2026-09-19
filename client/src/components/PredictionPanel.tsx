import { useMemo } from "react";

import type {
  Compatibility,
  InferencePayload,
  ModelLoadedPayload,
} from "../types";
import { HELP, TRAIN_HELP } from "../helpText";
import { ClassSpikeBarsFromInference } from "./ClassSpikeBarsFromInference";
import { HelpTip } from "./HelpTip";
import { PanelHeader } from "./PanelHeader";
import { confidenceSeries } from "./readoutMath";

interface Props {
  inference: InferencePayload | null;
  /** Shared time cursor step; drives the cumulative output bars. */
  timeStep: number | null;
  loaded: ModelLoadedPayload | null;
  /** Whether auto-prediction is switched on. */
  autoPredict: boolean;
  /** Whether a prediction can run at all (model + connection). */
  canAutoPredict: boolean;
  onToggleAutoPredict: () => void;
}

function MismatchBanner({ compatibility }: { compatibility: Compatibility }) {
  const legacy = compatibility.expected_input_mode === "raw";
  return (
    <div className="mismatch">
      <div className="mismatch-title">
        ⚠ Checkpoint mismatch
        <HelpTip text={TRAIN_HELP.compatibility} />
      </div>
      {legacy ? (
        <div>Legacy raw checkpoint — the encoding controls are ignored.</div>
      ) : (
        <div>
          Trained with “{compatibility.expected_input_mode}”; current is “
          {compatibility.current_coding}”.
        </div>
      )}
      {!compatibility.dataset_match && (
        <div>Dataset differs — prediction is disabled until they match.</div>
      )}
      {!compatibility.num_steps_match && <div>Time-step count differs.</div>}
    </div>
  );
}

/** One-line result: verdict mark, input label, prediction, confidence. */
function PredictionReadout({
  inference,
  confidencePct,
}: {
  inference: InferencePayload;
  /** Confidence to show; follows the cursor while it is active. */
  confidencePct: number;
}) {
  const correct =
    inference.true_label === null ||
    inference.true_label === inference.predicted;
  return (
    <div className="pred-row" data-testid="prediction-readout">
      <span
        className={`pred-mark ${correct ? "ok" : "bad"}`}
        title={correct ? "match" : "mismatch"}
      >
        {correct ? "✓" : "✗"}
      </span>
      <span>
        Input: <b>{inference.true_label ?? "—"}</b>
      </span>
      <span>
        Prediction: <b data-testid="prediction-value">{inference.predicted}</b>
      </span>
      <span>
        Confidence: <b>{confidencePct.toFixed(1)}%</b>
      </span>
    </div>
  );
}

/** The viewer tab's prediction card: live readout for the displayed sample. */
export function PredictionPanel({
  inference,
  timeStep,
  loaded,
  autoPredict,
  canAutoPredict,
  onToggleAutoPredict,
}: Props) {
  // Per-step confidence trace, so the readout can count up/down with the
  // cursor instead of always showing the final value.
  const confSeries = useMemo(
    () =>
      inference && inference.output_over_time.length > 1
        ? confidenceSeries(inference.output_over_time)
        : null,
    [inference],
  );
  const cursorConfidence =
    confSeries && timeStep !== null && timeStep !== undefined
      ? confSeries[Math.max(0, Math.min(timeStep, confSeries.length - 1))]
      : null;

  const compatibility = loaded?.compatibility;
  const mismatch =
    compatibility !== undefined &&
    compatibility !== null &&
    (!compatibility.coding_match ||
      !compatibility.dataset_match ||
      !compatibility.num_steps_match ||
      compatibility.expected_input_mode === "raw");

  return (
    <>
      {mismatch && compatibility && (
        <MismatchBanner compatibility={compatibility} />
      )}

      <div className="panel displayed-sample" data-testid="prediction-panel">
        <PanelHeader
          title="Prediction (displayed sample)"
          hint={HELP.inference}
          actions={
            <label
              className="toggle"
              title="Auto-predict the displayed sample"
            >
              <input
                type="checkbox"
                checked={autoPredict}
                onChange={onToggleAutoPredict}
                data-testid="auto-predict"
                aria-label="Auto-predict the displayed sample"
              />
              <span className="toggle-track" aria-hidden="true" />
            </label>
          }
        />
        {autoPredict && inference ? (
          <>
            <PredictionReadout
              inference={inference}
              confidencePct={cursorConfidence ?? inference.confidence * 100}
            />
            {!inference.dataset_match && (
              <div className="mismatch">
                Inference dataset differs from the checkpoint's training
                dataset.
              </div>
            )}
            <ClassSpikeBarsFromInference
              inference={inference}
              timeStep={timeStep}
            />
          </>
        ) : (
          <div className="muted">
            {!autoPredict
              ? "Auto-predict is off."
              : canAutoPredict
                ? "Waiting for the model to score the sample…"
                : "Train or load a model to see predictions."}
          </div>
        )}
      </div>
    </>
  );
}
