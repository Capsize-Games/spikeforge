export type CodingType = "rate" | "latency" | "delta" | "random";

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
}

export interface TrainStatePayload {
  running: boolean;
  reason: string;
}

export interface PredictionPayload {
  digits: number[];
  labels: number[];
}

export interface ModelListPayload {
  models: SavedModel[];
  datasets: DatasetInfo[];
}

export interface ModelLoadedPayload {
  name: string;
  dataset: string;
  accuracy: number;
}

export type ServerMsg =
  | { type: "config_ack"; payload: EncodeConfig }
  | { type: "status"; payload: StatusPayload | string }
  | { type: "image"; payload: number[][]; kind?: string }
  | { type: "raster"; payload: RasterPayload }
  | { type: "spike_frame"; payload: number[][]; step?: number }
  | { type: "run_state"; payload: RunStatePayload }
  | { type: "train_metrics"; payload: TrainMetrics }
  | { type: "train_state"; payload: TrainStatePayload }
  | { type: "prediction"; payload: PredictionPayload }
  | { type: "model_saved"; payload: { name: string; path: string } }
  | { type: "model_list"; payload: ModelListPayload }
  | { type: "model_loaded"; payload: ModelLoadedPayload }
  | { type: "error"; payload: string };
