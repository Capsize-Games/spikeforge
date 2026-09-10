/** Shared protocol types for model-zoo payloads and execution mode.

These live in their own module so `types.ts` stays within the 250-line limit;
`types.ts` re-exports them, so existing importers are unaffected.
*/

/**
 * Execution mode: production skips trajectory capture, educational keeps it.
 */
export type ExecutionMode = "educational" | "production";

/** Data modality a dataset delivers: static images or address-events. */
export type Modality = "image" | "event";

export interface DatasetInfo {
  name: string;
  classes: number;
  description: string;
  /** Additive registry metadata: static image vs neuromorphic events. */
  modality?: Modality;
  /** False when the dataset's loader is missing (e.g. no tonic extra). */
  available?: boolean;
}

export interface SavedModel {
  name: string;
  meta: Record<string, unknown>;
  saved_at: number;
}

export interface ModelListPayload {
  models: SavedModel[];
  datasets: DatasetInfo[];
  /** Selectable topology registry names. */
  topologies: string[];
  /** Selectable neuron kinds. */
  neurons: string[];
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
  mode?: ExecutionMode;
  device?: string;
  meta?: Record<string, unknown>;
  compatibility?: Compatibility;
}

/** The reproducibility record persisted beside a checkpoint. */
export interface ReproducibilityManifest {
  schema_version: number;
  created_at: number;
  config_hash: string;
  seed: number | null;
  versions: Record<string, string | null>;
  config: Record<string, unknown>;
  history: HistoryPoint[];
  reproducible: {
    bit_exact: boolean;
    guaranteed: string[];
    not_guaranteed: string[];
  };
  /** False for a legacy checkpoint with no stored manifest. */
  available?: boolean;
  reason?: string;
}

/** One registry record returned by a model search. */
export interface SearchRecord extends SavedModel {
  input_mode?: string;
  coding?: string;
  topology?: string;
  accuracy: number | null;
  manifest: ReproducibilityManifest | null;
}

export interface ModelSearchPayload {
  models: SearchRecord[];
}

/** One classified metadata key in a checkpoint diff. */
export interface DiffEntry {
  key: string;
  left: unknown;
  right: unknown;
  status: "added" | "removed" | "changed" | "same";
}

export interface ModelDiffPayload {
  left: { name: string | null; meta: Record<string, unknown> };
  right: { name: string | null; meta: Record<string, unknown> };
  entries: DiffEntry[];
  changed: string[];
  added: string[];
  removed: string[];
  identical: boolean;
  config_hash: {
    left: string | null;
    right: string | null;
    matches: boolean;
  };
}

/** Observable status of the opt-in training scale-ups (all default off). */
export interface ScaleUpStatus {
  amp: boolean;
  amp_dtype: string | null;
  grad_checkpoint: boolean;
  bptt_steps: number | null;
  multi_gpu: boolean;
  multi_gpu_status: string;
}

export type ModelRegistryServerMsg =
  | { type: "model_search"; payload: ModelSearchPayload }
  | { type: "model_diff"; payload: ModelDiffPayload };

/** Additive in-process metrics snapshot surfaced with system stats. */
export interface MetricsSnapshot {
  counters: Record<string, number>;
  gauges: Record<string, number>;
  timers: Record<string, Record<string, number>>;
}
