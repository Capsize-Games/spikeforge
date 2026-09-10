/** Protocol payloads for the Phase 2 introspection actions. */

/** Caps applied to a trajectory payload so it stays small. */
export interface TrajectoryCaps {
  max_neurons: number;
  max_stages: number;
}

/**
 * One stage's bounded traces: `[T][N]` rows for membrane `U[t]`, input
 * current `I[t]`, and spikes `S[t]`, plus the traced feature width.
 */
export interface StageTrace {
  neurons: number;
  membrane: number[][];
  current: number[][];
  spikes: number[][];
}

/** Payload emitted by the `trajectory` action. */
export interface TrajectoryPayload {
  steps: number;
  stages: string[];
  caps: TrajectoryCaps;
  traces: Record<string, StageTrace>;
}

/** Inter-spike-interval statistics; null when undefined. */
export interface IsiStats {
  count: number;
  mean: number | null;
  median: number | null;
  std: number | null;
  cv: number | null;
}

/** A firing-rate histogram: bin edges and per-bin counts. */
export interface FiringRateHistogram {
  edges: number[];
  counts: number[];
}

/** Metrics for one stage within the `metrics` payload. */
export interface StageMetrics {
  firing_rate: number;
  sparsity: number;
  isi: IsiStats;
  histogram: FiringRateHistogram;
}

/** Payload emitted by the `metrics` action. */
export interface TrajectoryMetricsPayload {
  steps: number;
  stages: Record<string, StageMetrics>;
}

/** Payload emitted by the `encoding_report` action. */
export interface EncodingReportPayload {
  coding: string;
  num_steps: number;
  shape: number[];
  firing_rate: number;
  sparsity: number;
  reconstruction: number[][][] | null;
  reconstruction_supported: boolean;
  approximation: string;
  stats: Record<string, number | string | boolean | null>;
}

/** Payload emitted by the `surrogate_curve` action. */
export interface SurrogateCurvePayload {
  name: string;
  x: number[];
  y: number[];
}

/** One timing block (forward or backward) in a benchmark record. */
export interface BenchmarkTiming {
  total_ms: number;
  median_total_ms: number;
  mean_ms_per_step: number;
  median_ms_per_step: number;
  steps_per_second: number;
  repeats: number;
}

/** Memory measurement block in a benchmark record. */
export interface BenchmarkMemory {
  cuda_peak_bytes: number | null;
  process_rss_bytes: number | null;
  tracemalloc_peak_bytes: number | null;
}

/** One topology/mode record in a benchmark report. */
export interface BenchmarkRecord {
  topology: string;
  mode: string;
  compiled: boolean;
  compile_status: string;
  forward: BenchmarkTiming;
  backward: BenchmarkTiming | null;
  memory: BenchmarkMemory;
}

/** Torch build and resolved device in a benchmark report. */
export interface BenchmarkEnvironment {
  torch_version: string;
  device: string;
  cuda_available: boolean;
  cuda_device: string | null;
  compile_available: boolean;
}

/** The resolved fixture echoed back in a benchmark report. */
export interface BenchmarkConfigReport {
  topologies: string[];
  batch_size: number;
  steps: number;
  repeats: number;
  warmup: number;
  seed: number;
  device: string;
  compiled: boolean;
  backward: boolean;
}

/** Payload emitted by the `benchmark` action. */
export interface BenchmarkPayload {
  config: BenchmarkConfigReport;
  environment: BenchmarkEnvironment;
  results: BenchmarkRecord[];
}

/** Server message variants added by the introspection actions. */
export type IntrospectionServerMsg =
  | { type: "trajectory"; payload: TrajectoryPayload }
  | { type: "metrics"; payload: TrajectoryMetricsPayload }
  | { type: "encoding_report"; payload: EncodingReportPayload }
  | { type: "surrogate_list"; payload: string[] }
  | { type: "surrogate_curve"; payload: SurrogateCurvePayload }
  | { type: "benchmark"; payload: BenchmarkPayload };
