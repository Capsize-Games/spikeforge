import type { CodingType, DeviceChoice } from "./configTypes";
import type {
  ExecutionMode,
  MetricsSnapshot,
  Modality,
  ScaleUpStatus,
} from "./protocolTypes";

/** Which layer produced a raster / spike frame. */
export type RasterSource = "input" | "hidden" | "output";

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
  /** Public demo mode disables training and model changes. */
  read_only?: boolean;
}

export interface RunStatePayload {
  running: boolean;
  reason: string;
}

/** Whether the opt-in hidden-layer animation stream is available. */
export interface AnimationStatePayload {
  available: boolean;
  /** Named reason the animation is unavailable; empty when available. */
  reason: string;
  source: RasterSource;
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
