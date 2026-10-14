import { useEffect, useState } from "react";

import type { BenchmarkPayload } from "./introspectionTypes";
import { useTrainingMessages } from "./trainingMessages";
import type {
  DatasetInfo,
  ModelLoadedPayload,
  PredictionPayload,
  SavedModel,
  TrainConfig,
  TrainMetrics,
} from "./types";
import { readJson, STORAGE_KEYS, writeJson } from "./storage";
import { defaultConfig, defaultTrainConfig } from "./types";

export interface TrainingState {
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

  const handleMessage = useTrainingMessages(setState);

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
