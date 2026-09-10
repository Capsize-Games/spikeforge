import { useCallback, useState } from "react";

import type {
  DatasetInfo,
  ModelLoadedPayload,
  PredictionPayload,
  SavedModel,
  ServerMsg,
  TrainConfig,
  TrainMetrics,
} from "./types";
import { defaultTrainConfig } from "./types";

interface TrainingState {
  config: TrainConfig;
  running: boolean;
  loss: number[];
  testAccuracy: number[];
  trainAccuracy: number[];
  last: TrainMetrics | null;
  device: string | null;
  prediction: PredictionPayload | null;
  models: SavedModel[];
  datasets: DatasetInfo[];
  loaded: ModelLoadedPayload | null;
  status: string | null;
}

const initial: TrainingState = {
  config: defaultTrainConfig,
  running: false,
  loss: [],
  testAccuracy: [],
  trainAccuracy: [],
  last: null,
  device: null,
  prediction: null,
  models: [],
  datasets: [],
  loaded: null,
  status: null,
};

/** Track training config, live metrics, and saved models. */
export function useTraining() {
  const [state, setState] = useState<TrainingState>(initial);

  const handleMessage = useCallback((msg: ServerMsg): boolean => {
    switch (msg.type) {
      case "train_metrics": {
        const m = msg.payload;
        setState((s) => ({
          ...s,
          last: m,
          device: m.device ?? s.device,
          loss: [...s.loss, m.loss],
          trainAccuracy: [...s.trainAccuracy, m.train_accuracy * 100],
          testAccuracy:
            m.test_accuracy !== null
              ? [...s.testAccuracy, m.test_accuracy]
              : s.testAccuracy,
        }));
        return true;
      }
      case "train_state":
        setState((s) => ({
          ...s,
          running: msg.payload.running,
          device: msg.payload.device ?? s.device,
        }));
        return true;
      case "prediction":
        setState((s) => ({ ...s, prediction: msg.payload }));
        return true;
      case "model_list":
        setState((s) => ({
          ...s,
          models: msg.payload.models,
          datasets: msg.payload.datasets,
        }));
        return true;
      case "model_loaded": {
        const h = msg.payload.history ?? [];
        setState((s) => ({
          ...s,
          loaded: msg.payload,
          loss: h.map((p) => p.loss),
          trainAccuracy: h.map((p) => p.train_accuracy * 100),
          testAccuracy: h
            .filter((p) => p.test_accuracy !== null)
            .map((p) => p.test_accuracy as number),
          last: h.length ? { ...h[h.length - 1], total: h.length } : s.last,
        }));
        return true;
      }
      case "model_cleared":
        setState((s) => ({
          ...s,
          loaded: null,
          prediction: null,
          config: { ...s.config, checkpoint: null },
        }));
        return true;
      case "model_saved":
        setState((s) => ({ ...s, status: `saved ${msg.payload.name}` }));
        return true;
      default:
        return false;
    }
  }, []);

  const patch = (patchConfig: Partial<TrainConfig>) =>
    setState((s) => ({ ...s, config: { ...s.config, ...patchConfig } }));

  const reset = () =>
    setState((s) => ({
      ...s,
      loss: [],
      trainAccuracy: [],
      testAccuracy: [],
      last: null,
      prediction: null,
    }));

  const setRunning = (running: boolean) =>
    setState((s) => ({ ...s, running }));

  return { state, handleMessage, patch, reset, setRunning };
}
