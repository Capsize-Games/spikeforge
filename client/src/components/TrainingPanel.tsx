import type {
  Compatibility,
  InferencePayload,
  ModelLoadedPayload,
  PredictionPayload,
  TrainMetrics,
} from "../types";
import { TRAIN_HELP } from "../helpText";
import { ClassSpikeBarsFromInference } from "./ClassSpikeBars";
import { HelpTip } from "./HelpTip";
import { LineChart } from "./LineChart";

interface Props {
  loss: number[];
  trainAccuracy: number[];
  testAccuracy: number[];
  last: TrainMetrics | null;
  device: string | null;
  inference: InferencePayload | null;
  prediction: PredictionPayload | null;
  loaded: ModelLoadedPayload | null;
}

function MismatchBanner({ compatibility }: { compatibility: Compatibility }) {
  const legacy = compatibility.expected_input_mode === "raw";
  return (
    <div className="mismatch">
      <div className="mismatch-title">⚠ Checkpoint mismatch</div>
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

export function TrainingPanel({
  loss,
  trainAccuracy,
  testAccuracy,
  last,
  device,
  inference,
  prediction,
  loaded,
}: Props) {
  const compatibility = loaded?.compatibility;
  const mismatch =
    compatibility !== undefined &&
    compatibility !== null &&
    (!compatibility.coding_match ||
      !compatibility.dataset_match ||
      !compatibility.num_steps_match ||
      compatibility.expected_input_mode === "raw");

  return (
    <div className="col-training">
      {mismatch && compatibility && (
        <MismatchBanner compatibility={compatibility} />
      )}

      {loaded && (
        <div className="panel">
          <div className="panel-title">
            <span>Loaded model</span>
            <HelpTip text={TRAIN_HELP.input_mode} />
          </div>
          <div className="metrics">
            <div>{loaded.name}</div>
            <div>{loaded.dataset}</div>
            <div>acc {loaded.accuracy.toFixed(1)}%</div>
            <div>input {loaded.input_mode ?? "raw"}</div>
            <div>hidden {loaded.hidden ?? "—"}</div>
            <div>device {loaded.device ?? "—"}</div>
          </div>
        </div>
      )}

      <div className="panel displayed-sample">
        <div className="panel-title">Prediction (displayed sample)</div>
        {inference ? (
          <>
            <div className="metrics">
              <div>predicted {inference.predicted}</div>
              <div>confidence {(inference.confidence * 100).toFixed(1)}%</div>
              <div>
                true {inference.true_label !== null ? inference.true_label : "—"}
              </div>
              <div>
                {inference.true_label === null ||
                inference.true_label === inference.predicted
                  ? "match ✓"
                  : "mismatch ✗"}
              </div>
            </div>
            {!inference.dataset_match && (
              <div className="mismatch">
                Inference dataset differs from the checkpoint's training dataset.
              </div>
            )}
            <ClassSpikeBarsFromInference inference={inference} />
          </>
        ) : prediction ? (
          <div className="predictions">
            {prediction.digits.map((d, i) => (
              <span
                key={i}
                className={d === prediction.labels[i] ? "pred ok" : "pred bad"}
              >
                {d}/{prediction.labels[i]}
              </span>
            ))}
          </div>
        ) : (
          <div className="muted">
            Expand section 4 and click “Predict displayed sample”.
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-title">Training</div>

        <div className="panel-title subsection">Status</div>
        {last ? (
          <div className="metrics">
            <div>step {last.step} / {last.total}</div>
            <div>epoch {last.epoch + 1}</div>
            <div>loss {last.loss.toFixed(3)}</div>
            <div>
              held-out{" "}
              {last.test_accuracy !== null
                ? `${last.test_accuracy.toFixed(1)}%`
                : "—"}
            </div>
            <div>batch {(last.train_accuracy * 100).toFixed(1)}%</div>
            <div>device {device ?? last.device ?? "—"}</div>
            <div>
              {last.step_ms !== undefined
                ? `${last.step_ms.toFixed(0)} ms/step`
                : "— ms/step"}
            </div>
          </div>
        ) : (
          <div className="muted">Not training</div>
        )}

        <div className="panel-title subsection">Loss (training)</div>
        <LineChart
          bare
          series={[{ label: "loss", color: "#f85149", values: loss }]}
        />

        <div className="panel-title subsection">Accuracy (%)</div>
        <LineChart
          bare
          series={[
            {
              label: "held-out",
              color: "#3fb950",
              values: testAccuracy,
              max: 100,
            },
            {
              label: "train batch",
              color: "#8b949e",
              values: trainAccuracy,
              max: 100,
            },
          ]}
        />
      </div>
    </div>
  );
}
