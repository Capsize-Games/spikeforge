import type { IntrospectionServerMsg } from "./introspectionTypes";
import type { NirGraphPayload, NirValidationPayload } from "./nirTypes";
import type {
  ExecutionMode,
  MetricsSnapshot,
  Modality,
  ModelLoadedPayload,
  ModelListPayload,
  ModelRegistryServerMsg,
  ScaleUpStatus,
} from "./protocolTypes";
import type { TargetServerMsg } from "./targetTypes";

export type {
  Compatibility,
  DatasetInfo,
  ExecutionMode,
  HistoryPoint,
  Modality,
  ModelLoadedPayload,
  ModelListPayload,
  SavedModel,
  ScaleUpStatus,
} from "./protocolTypes";

export type CodingType = "rate" | "latency" | "delta" | "random";

/** Compute device selectable for training. */
export type DeviceChoice = "auto" | "cpu" | "gpu";

/** Per-topology overrides for the registry defaults. */
export type TopologyParams = Record<string, number | string | boolean>;

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
  /** Modality of the configured sample (additive). */
  modality?: Modality;
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
  /** Execution mode for the run. */
  mode: ExecutionMode;
  /** Selectable topology registry name. */
  topology: string;
  /** Per-topology overrides for the registry defaults. */
  topology_params: TopologyParams;
  /** Opt-in mixed precision; default false preserves fp32 numerics. */
  amp: boolean;
  /** Opt-in per-step activation checkpointing; default false. */
  grad_checkpoint: boolean;
  /** Truncated-BPTT window; null means full backprop-through-time. */
  bptt_steps: number | null;
  /** Opt-in multi-GPU DataParallel; default false. */
  multi_gpu: boolean;
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
  mode: "production",
  topology: "fc_legacy",
  topology_params: {},
  amp: false,
  grad_checkpoint: false,
  bptt_steps: null,
  multi_gpu: false,
  encode: { ...defaultConfig },
};

export interface TrainMetrics {
  loss: number;
  train_accuracy: number;
  test_accuracy: number | null;
  epoch: number;
  step: number;
  total: number;
  device?: string;
  step_ms?: number;
  /** Observable status of the opt-in training scale-ups. */
  scaleup?: ScaleUpStatus;
}

export interface TrainStatePayload {
  running: boolean;
  reason: string;
  device?: string;
  mode?: ExecutionMode;
  /** Present on the terminal (finished/error) training state. */
  scaleup?: ScaleUpStatus;
}

export interface PredictionPayload {
  digits: number[];
  labels: number[];
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
  /** Additive in-process metrics snapshot (Phase 6c); may be absent. */
  metrics?: MetricsSnapshot;
}

/** Dataset-download progress streamed from the server. */
export interface DownloadState {
  dataset: string;
  status: "idle" | "downloading" | "done" | "cancelled" | "error";
  bytes: number;
}

export type ServerMsg =
  | { type: "config_ack"; payload: EncodeConfig & { modality?: Modality } }
  | { type: "status"; payload: StatusPayload | string }
  | { type: "image"; payload: number[][]; kind?: string }
  | { type: "raster"; payload: RasterPayload; source?: RasterSource }
  | { type: "spike_frame"; payload: number[][]; step?: number;
      source?: RasterSource }
  | { type: "run_state"; payload: RunStatePayload }
  | { type: "train_metrics"; payload: TrainMetrics }
  | { type: "train_state"; payload: TrainStatePayload }
  | { type: "prediction"; payload: PredictionPayload }
  | { type: "inference"; payload: InferencePayload }
  | { type: "nir_graph"; payload: NirGraphPayload }
  | { type: "nir_validation"; payload: NirValidationPayload }
  | { type: "model_saved"; payload: { name: string; path: string } }
  | { type: "model_list"; payload: ModelListPayload }
  | { type: "model_loaded"; payload: ModelLoadedPayload }
  | { type: "model_cleared"; payload?: null }
  | { type: "system_stats"; payload: SystemStatsPayload }
  | { type: "download_state"; payload: DownloadState }
  | IntrospectionServerMsg
  | TargetServerMsg
  | ModelRegistryServerMsg
  | { type: "error"; payload: string };
