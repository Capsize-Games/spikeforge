import { useCallback } from "react";
import type { Dispatch, SetStateAction } from "react";

import type {
  ModelLoadedPayload,
  ServerMsg,
  TrainMetrics,
} from "./types";
import type { TrainingState } from "./useTraining";

/** Fields derived from a checkpoint's stored history on load. */
function loadedState(
  payload: ModelLoadedPayload,
  prevLast: TrainMetrics | null,
): Partial<TrainingState> {
  const h = payload.history ?? [];
  return {
    loaded: payload,
    loss: h.map((p) => p.loss),
    trainAccuracy: h.map((p) => p.train_accuracy * 100),
    testAccuracy: h
      .filter((p) => p.test_accuracy !== null)
      .map((p) => p.test_accuracy as number),
    last: h.length ? { ...h[h.length - 1], total: h.length } : prevLast,
  };
}

/** Fold training server messages into the training state setter. */
export function useTrainingMessages(
  setState: Dispatch<SetStateAction<TrainingState>>,
) {
  const handleMessage = useCallback(
    (msg: ServerMsg): boolean => {
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
          setState((s) => {
            // A checkpoint name can persist in localStorage across sessions
            // (e.g. after the server's model directory was cleared or
            // rebuilt); if it's no longer in the server's own list, drop it
            // so the next plain "Train" doesn't try to resume a checkpoint
            // that no longer exists.
            const stale =
              s.config.checkpoint !== null &&
              !msg.payload.models.some(
                (m) => m.name === s.config.checkpoint,
              );
            return {
              ...s,
              models: msg.payload.models,
              datasets: msg.payload.datasets,
              topologies: msg.payload.topologies,
              neurons: msg.payload.neurons,
              config: stale ? { ...s.config, checkpoint: null } : s.config,
            };
          });
          return true;
        case "surrogate_list":
          setState((s) => ({ ...s, surrogates: msg.payload }));
          return true;
        case "benchmark":
          setState((s) => ({
            ...s,
            benchmark: msg.payload,
            benchmarkLoading: false,
          }));
          return true;
        case "error":
          // A failed run must not leave the benchmark spinner stuck; the
          // viewer still owns the visible error banner, so return false.
          setState((s) =>
            s.benchmarkLoading ? { ...s, benchmarkLoading: false } : s,
          );
          return false;
        case "model_loaded":
          setState((s) => ({ ...s, ...loadedState(msg.payload, s.last) }));
          return true;
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
    },
    [setState],
  );
  return handleMessage;
}
