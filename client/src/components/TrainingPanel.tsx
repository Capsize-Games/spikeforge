import type {
  EncodeConfig,
  InferencePayload,
  ModelLoadedPayload,
  PredictionPayload,
  SavedModel,
  SystemStatsPayload,
  TrainConfig,
  TrainMetrics,
} from "../types";
import { ClassSpikeBarsFromInference } from "./ClassSpikeBars";
import { HelpTip } from "./HelpTip";
import { LineChart } from "./LineChart";
import { ResourceMonitor } from "./ResourceMonitor";
import { TrainControls } from "./TrainControls";
import { TRAIN_HELP } from "../helpText";

interface Props {
  config: TrainConfig;
  encode: EncodeConfig;
  models: SavedModel[];
  onChange: (patch: Partial<TrainConfig>) => void;
  onTrain: () => void;
  onStop: () => void;
  onInfer: () => void;
  onSave: (name: string) => void;
  onLoad: (name: string) => void;
  onDelete: (name: string) => void;
  running: boolean;
  connected: boolean;
  canInfer: boolean;
  stats: SystemStatsPayload | null;
  gpuAvailable: boolean;
  loss: number[];
  trainAccuracy: number[];
  testAccuracy: number[];
  last: TrainMetrics | null;
  inference: InferencePayload | null;
  prediction: PredictionPayload | null;
  loaded: ModelLoadedPayload | null;
}

export function TrainingPanel({
  config,
  encode,
  models,
  onChange,
  onTrain,
  onStop,
  onInfer,
  onSave,
  onLoad,
  onDelete,
  running,
  connected,
  canInfer,
  stats,
  gpuAvailable,
  loss,
  trainAccuracy,
  testAccuracy,
  last,
  inference,
  prediction,
  loaded,
}: Props) {
  return (
    <div className="col-training">
      <TrainControls
        config={config}
        encode={encode}
        models={models}
        compatibility={loaded?.compatibility}
        onChange={onChange}
        onTrain={onTrain}
        onStop={onStop}
        onInfer={onInfer}
        onSave={onSave}
        onLoad={onLoad}
        onDelete={onDelete}
        running={running}
        connected={connected}
        canInfer={canInfer}
        gpuAvailable={gpuAvailable}
      />

      <ResourceMonitor stats={stats} requested={config.device} />

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
            <div>coding {loaded.coding ?? "raw"}</div>
            {loaded.hidden !== undefined && <div>hidden {loaded.hidden}</div>}
            {loaded.num_steps !== undefined && <div>steps {loaded.num_steps}</div>}
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
          <div className="muted">Run "Infer on this sample"</div>
        )}
      </div>

      <LineChart
        title="Loss (training)"
        series={[{ label: "loss", color: "#f85149", values: loss }]}
      />
      <LineChart
        title="Accuracy (%)"
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

      <div className="panel">
        <div className="panel-title">Status</div>
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
          </div>
        ) : (
          <div className="muted">Not training</div>
        )}
      </div>
    </div>
  );
}
