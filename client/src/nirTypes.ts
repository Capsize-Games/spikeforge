/** Protocol payloads for the NIR export and validation actions. */

/** One node in an exported NIR graph summary. */
export interface NirGraphNode {
  name: string;
  kind: string;
  params: Record<string, unknown>;
}

/** One directed edge in an exported NIR graph summary. */
export interface NirGraphEdge {
  source: string;
  target: string;
  delayed: boolean;
}

/** Node/edge listing emitted by the `nir_export` action. */
export interface NirGraphPayload {
  nodes: NirGraphNode[];
  edges: NirGraphEdge[];
}

/** Error metrics for one compared layer quantity. */
export interface DriftMetrics {
  max_abs: number;
  mean_abs: number;
  relative: number;
  agreement: number;
}

/** Per-layer comparison result within a validation report. */
export interface NirValidationLayer {
  within_tolerance: boolean;
  spikes?: DriftMetrics;
  membrane?: DriftMetrics;
}

/** Descriptor naming the worst offending layer quantity. */
export interface NirValidationWorst {
  layer: string;
  quantity: string;
  metric: string;
  value: number;
}

/** Drift report emitted by the `nir_validate` action. */
export interface NirValidationPayload {
  within_tolerance: boolean;
  steps: number;
  layers: Record<string, NirValidationLayer>;
  readout: DriftMetrics;
  worst: NirValidationWorst | null;
  notes: string[];
}
