/** Protocol payloads and client-only state for the Pipeline tab. */

import type { PipelineGraph } from "./protocol/generated";

export type { PipelineGraph };

/** One saved pipeline's summary, as listed (not its full graph). */
export interface PipelineSummary {
  name: string;
  node_count: number;
  edge_count: number;
  saved_at: number;
}

/** Payload emitted by the `list_pipelines` action. */
export interface PipelineListPayload {
  pipelines: PipelineSummary[];
}

/** One node's result, in the same shape a single-model prediction takes. */
export interface PipelineNodeResult {
  steps: number;
  label: number;
  predicted: number;
  mean_logits: { dtype: string; shape: number[]; values: number[][] };
  [key: string]: unknown;
}

/** Payload emitted per node as a run progresses. */
export interface PipelineNodeResultPayload {
  node_id: string;
  result: PipelineNodeResult;
}

/** Payload emitted when a run starts, finishes, stops, or errors. */
export interface PipelineRunStatePayload {
  running: boolean;
  reason?: "finished" | "stopped" | "error";
}

export type PipelineServerMsg =
  | { type: "pipeline_list"; payload: PipelineListPayload }
  | { type: "pipeline_saved"; payload: { name: string } }
  | { type: "pipeline_loaded"; payload: PipelineGraph }
  | {
      type: "pipeline_deleted";
      payload: { name: string; existed: boolean };
    }
  | { type: "pipeline_node_result"; payload: PipelineNodeResultPayload }
  | { type: "pipeline_run_state"; payload: PipelineRunStatePayload };

/** A node's live status in the canvas, driven by run_state/node_result. */
export type PipelineNodeStatus = "idle" | "running" | "done" | "error";

/** The raw run request feeding a pipeline's source node(s). */
export interface PipelineRunInput {
  frames: unknown[];
  encoded: boolean;
}
