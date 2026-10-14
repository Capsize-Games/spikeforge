import type { ExecutionMode } from "./protocolTypes";

export type CodingType = "rate" | "latency" | "delta" | "random";

/** Compute device selectable for training. */
export type DeviceChoice = "auto" | "cpu" | "gpu";

/** Per-stage parameter overrides keyed by stage name. */
export type StageParams = Record<string, number | string | boolean>;

/** Per-stage neuron-kind overrides keyed by stage name. */
export type StageNeurons = Record<string, string>;

/** One topology override value: a scalar or a nested stage mapping. */
export type TopologyValue =
  | number
  | string
  | boolean
  | Record<string, number | string | boolean>;

/** Per-topology overrides for the registry defaults. */
export type TopologyParams = Record<string, TopologyValue>;

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
  /** Optional (H, W) sensor geometry; null keeps the default 28x28. */
  input_size: [number, number] | null;
  /** Opt-in per-step hidden-layer animation; default off. */
  animate_hidden: boolean;
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
  input_size: null,
  animate_hidden: false,
};

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
  /** Per-stage neuron-kind overrides, additive and default-empty. */
  stage_neurons: StageNeurons;
  /** Per-stage parameter overrides, additive and default-empty. */
  stage_params: StageParams;
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
  stage_neurons: {},
  stage_params: {},
  amp: false,
  grad_checkpoint: false,
  bptt_steps: null,
  multi_gpu: false,
  encode: { ...defaultConfig },
};
