/** Shared protocol types for model-zoo payloads and execution mode.

These live in their own module so `types.ts` stays within the 250-line limit;
`types.ts` re-exports them, so existing importers are unaffected.
*/

/**
 * Execution mode: production skips trajectory capture, educational keeps it.
 */
export type ExecutionMode = "educational" | "production";

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
