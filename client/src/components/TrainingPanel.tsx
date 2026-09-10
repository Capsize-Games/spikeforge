import type {
  DatasetInfo,
  ModelLoadedPayload,
  PredictionPayload,
  SavedModel,
  TrainConfig,
  TrainMetrics,
} from "../types";
import { LineChart } from "./LineChart";
import { TrainControls } from "./TrainControls";

interface Props {
  config: TrainConfig;
  datasets: DatasetInfo[];
  models: SavedModel[];
  onChange: (patch: Partial<TrainConfig>) => void;
  onTrain: () => void;
  onStop: () => void;
  onPredict: () => void;
  onSave: (name: string) => void;
  onLoad: (name: string) => void;
  onDelete: (name: string) => void;
  running: boolean;
  connected: boolean;
  loss: number[];
  trainAccuracy: number[];
  testAccuracy: number[];
  last: TrainMetrics | null;
  prediction: PredictionPayload | null;
  loaded: ModelLoadedPayload | null;
}

export function TrainingPanel({
  config,
  datasets,
  models,
  onChange,
  onTrain,
  onStop,
  onPredict,
  onSave,
  onLoad,
  onDelete,
  running,
  connected,
  loss,
  trainAccuracy,
  testAccuracy,
  last,
  prediction,
  loaded,
}: Props) {
  return (
    <div className="col-training">
      <TrainControls
        config={config}
        datasets={datasets}
        models={models}
        onChange={onChange}
        onTrain={onTrain}
        onStop={onStop}
        onPredict={onPredict}
        onSave={onSave}
        onLoad={onLoad}
        onDelete={onDelete}
        running={running}
        connected={connected}
      />

      {loaded && (
        <div className="panel">
          <div className="panel-title">Loaded model</div>
          <div className="metrics">
            <div>{loaded.name}</div>
            <div>{loaded.dataset}</div>
            <div>acc {loaded.accuracy.toFixed(1)}%</div>
          </div>
        </div>
      )}

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

      <div className="panel">
        <div className="panel-title">Predictions (pred / true)</div>
        {prediction ? (
          <div className="predictions">
            {prediction.digits.map((d, i) => (
              <span
                key={i}
                className={
                  d === prediction.labels[i] ? "pred ok" : "pred bad"
                }
              >
                {d}/{prediction.labels[i]}
              </span>
            ))}
          </div>
        ) : (
          <div className="muted">Run "Predict sample"</div>
        )}
      </div>
    </div>
  );
}
