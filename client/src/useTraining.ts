import { useCallback, useEffect, useState } from "react";

import type { BenchmarkPayload } from "./introspectionTypes";
import type {
  DatasetInfo,
  ModelLoadedPayload,
  PredictionPayload,
  SavedModel,
  ServerMsg,
  TrainConfig,
  TrainMetrics,
} from "./types";
import { readJson, STORAGE_KEYS, writeJson } from "./storage";
import { defaultConfig, defaultTrainConfig } from "./types";

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
  /** Registry names for the architecture pickers, empty until listed. */
  topologies: string[];
  neurons: string[];
  surrogates: string[];
  loaded: ModelLoadedPayload | null;
  status: string | null;
  /** Latest benchmark report; null until one is run. */
  benchmark: BenchmarkPayload | null;
  /** True while a benchmark run is in flight. */
  benchmarkLoading: boolean;
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
  topologies: [],
  neurons: [],
  surrogates: [],
  loaded: null,
  status: null,
  benchmark: null,
  benchmarkLoading: false,
};

/** Restore persisted training settings, filling in any missing defaults. */
function loadTrainConfig(): TrainConfig {
  const saved = readJson<Partial<TrainConfig>>(STORAGE_KEYS.train, {});
  return {
    ...defaultTrainConfig,
    ...saved,
    encode: { ...defaultConfig, ...(saved.encode ?? {}) },
  };
}

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

/** Track training config, live metrics, and saved models. */
export function useTraining() {
  const [state, setState] = useState<TrainingState>(() => ({
    ...initial,
    config: loadTrainConfig(),
  }));

  // Persist model settings so a page reload restores them.
  useEffect(() => {
    writeJson(STORAGE_KEYS.train, state.config);
  }, [state.config]);

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
          topologies: msg.payload.topologies,
          neurons: msg.payload.neurons,
        }));
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

  const setBenchmarkLoading = (benchmarkLoading: boolean) =>
    setState((s) => ({ ...s, benchmarkLoading }));

  return {
    state,
    handleMessage,
    patch,
    reset,
    setRunning,
    setBenchmarkLoading,
  };
}
