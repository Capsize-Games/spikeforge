export type CodingType = "rate" | "latency" | "delta" | "random";

/** Compute device selectable for training. */
export type DeviceChoice = "auto" | "cpu" | "gpu";

/** Which layer produced a raster / spike frame. */
export type RasterSource = "input" | "hidden" | "output";

export interface EncodeConfig {
  coding: CodingType;
  dataset: string;
  subset: number;
  batch_size: number;
  num_steps: number;
  sample_index: number;
  gain: number;
  vector_value: number;
  tau: number;
  threshold: number;
  clip: boolean;
  normalize: boolean;
  linear: boolean;
  off_spike: boolean;
  delta_threshold: number;
  random_scale: number;
  random_seed: number | null;
  interval_ms: number;
}

export const defaultConfig: EncodeConfig = {
  coding: "rate",
  dataset: "mnist",
  subset: 10,
  batch_size: 128,
  num_steps: 100,
  sample_index: 0,
  gain: 0.25,
  vector_value: 0.5,
  tau: 5,
  threshold: 0.01,
  clip: false,
  normalize: true,
  linear: true,
  off_spike: false,
  delta_threshold: 4,
  random_scale: 0.5,
  random_seed: null,
  interval_ms: 100,
};

export interface RasterPayload {
  time: number[];
  neurons: number[];
  num_steps: number;
  num_neurons: number;
}

export interface StatusPayload {
  coding: CodingType;
  num_steps: number;
  target: number | null;
  dataset?: string;
  sample_index?: number;
  true_label?: number | null;
}

export interface RunStatePayload {
  running: boolean;
  reason: string;
}

export interface TrainConfig {
  dataset: string;
  hidden: number;
  beta: number;
  lr: number;
  epochs: number;
  num_steps: number;
  subset: number;
  batch_size: number;
  checkpoint: string | null;
  device: DeviceChoice;
  /** Encoding settings used for spike-input training. */
  encode: EncodeConfig;
}

export const defaultTrainConfig: TrainConfig = {
  dataset: "mnist",
  hidden: 256,
  beta: 0.9,
  lr: 0.005,
  epochs: 3,
  num_steps: 25,
  subset: 10,
  batch_size: 64,
  checkpoint: null,
  device: "auto",
  encode: { ...defaultConfig },
};

export interface DatasetInfo {
  name: string;
  classes: number;
  description: string;
}

export interface SavedModel {
  name: string;
  meta: Record<string, unknown>;
  saved_at: number;
}

export interface TrainMetrics {
  loss: number;
  train_accuracy: number;
  test_accuracy: number | null;
  epoch: number;
  step: number;
  total: number;
  device?: string;
  step_ms?: number;
}

export interface TrainStatePayload {
  running: boolean;
  reason: string;
  device?: string;
}

export interface PredictionPayload {
  digits: number[];
  labels: number[];
}

export interface ModelListPayload {
  models: SavedModel[];
  datasets: DatasetInfo[];
}

/** How a loaded checkpoint relates to the current encoding controls. */
export interface Compatibility {
  dataset_match: boolean;
  coding_match: boolean;
  num_steps_match: boolean;
  expected_input_mode: string;
  current_coding: string;
}

/** One persisted training-metric sample. */
export interface HistoryPoint {
  step: number;
  epoch: number;
  loss: number;
  train_accuracy: number;
  test_accuracy: number | null;
}

export interface ModelLoadedPayload {
  name: string;
  dataset: string;
  accuracy: number;
  history?: HistoryPoint[];
  input_mode?: string;
  coding?: string;
  hidden?: number;
  beta?: number;
  num_steps?: number;
  num_classes?: number;
  device?: string;
  meta?: Record<string, unknown>;
  compatibility?: Compatibility;
}

/** Result of inferring on the currently displayed sample. */
export interface InferencePayload {
  predicted: number;
  confidence: number;
  true_label: number | null;
  class_spikes: number[];
  output_over_time: number[][];
  coding: string;
  input_mode: string;
  num_steps: number;
  dataset_match: boolean;
}

/** One memory pool's usage, in bytes. */
export interface MemoryBlock {
  total: number;
  used: number;
  available: number;
  percent: number;
  name?: string;
}

/** Device availability plus the engine's currently active device. */
export interface DeviceStatus {
  available: DeviceChoice[];
  gpu_name: string | null;
  active: string | null;
}

export interface SystemStatsPayload {
  cpu: MemoryBlock | null;
  gpu: MemoryBlock | null;
  device: DeviceStatus;
}

export type ServerMsg =
  | { type: "config_ack"; payload: EncodeConfig }
  | { type: "status"; payload: StatusPayload | string }
  | { type: "image"; payload: number[][]; kind?: string }
  | { type: "raster"; payload: RasterPayload; source?: RasterSource }
  | {
      type: "spike_frame";
      payload: number[][];
      step?: number;
      source?: RasterSource;
    }
  | { type: "run_state"; payload: RunStatePayload }
  | { type: "train_metrics"; payload: TrainMetrics }
  | { type: "train_state"; payload: TrainStatePayload }
  | { type: "prediction"; payload: PredictionPayload }
  | { type: "inference"; payload: InferencePayload }
  | { type: "model_saved"; payload: { name: string; path: string } }
  | { type: "model_list"; payload: ModelListPayload }
  | { type: "model_loaded"; payload: ModelLoadedPayload }
  | { type: "model_cleared"; payload?: null }
  | { type: "system_stats"; payload: SystemStatsPayload }
  | { type: "error"; payload: string };
