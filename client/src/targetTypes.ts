/** Protocol payloads for the deployment-target actions (Phase 5c). */

import type { NirValidationPayload } from "./nirTypes";

/** Whether a target is the in-process reference, a simulator, or hardware. */
export type TargetKind = "reference" | "simulator" | "hardware";

/** The declared shape of a target, as emitted by `TargetSpec.to_dict()`. */
export interface TargetSpecDict {
  name: string;
  kind: TargetKind;
  description: string;
  extra: string | null;
  supported: string[];
  constraints: Record<string, unknown>;
  substitutions: Record<string, string>;
}

/** A target spec plus its live availability and primitive count. */
export interface TargetSummary extends TargetSpecDict {
  available: boolean;
  supported_count: number;
}

/** Payload emitted by the `targets` action. */
export interface TargetListPayload {
  targets: TargetSummary[];
}

/** One node a target replaces with a primitive it does support. */
export interface SubstitutionRecord {
  name: string;
  primitive: string;
  substitute: string;
}

/** Node count per capability bucket, including the total. */
export interface DeploymentCounts {
  supported: number;
  unsupported: number;
  substituted: number;
  total: number;
}

/** Per-node classification of one graph against one target. */
export interface DeploymentNodes {
  supported: string[];
  unsupported: string[];
  substituted: SubstitutionRecord[];
  counts: DeploymentCounts;
}

/** One node a rewrite applied, skipped, or found unfixable. */
export interface RewriteRecord {
  node: string;
  from?: string;
  to?: string;
  primitive?: string;
  detail?: string;
  reason?: string;
}

/** Drift metrics carried by the substitution drift check. */
export interface RewriteDriftMetrics {
  max_abs: number;
  mean_abs: number;
  relative: number;
  agreement: number;
}

/** Aggregate spike drift over the shared spiking nodes. */
export interface RewriteSpikeSummary {
  nodes: string[];
  max_abs: number;
  mean_abs: number;
  agreement: number;
}

/** Post-rewrite drift of the rewritten graph against the original. */
export interface RewriteDrift {
  steps: number;
  readout: RewriteDriftMetrics;
  spikes: RewriteSpikeSummary;
  within_tolerance: boolean;
}

/** Bucket sizes of a rewrite report. */
export interface RewriteCounts {
  applied: number;
  skipped: number;
  unfixable: number;
}

/** The executed-substitution report attached to a deployment report. */
export interface RewritePayload {
  target: string;
  applied: RewriteRecord[];
  skipped: RewriteRecord[];
  unfixable: RewriteRecord[];
  rewritten: boolean;
  ready: boolean;
  counts: RewriteCounts;
  drift: RewriteDrift | null;
  error?: string;
}

/** Honest status of a backend run. */
export type BackendStatus = "ok" | "unavailable" | "error";

/** A backend result compared against the reference interpreter. */
export interface BackendCompare {
  readout: RewriteDriftMetrics;
  spikes: RewriteSpikeSummary;
}

/** Payload emitted by the `deployment_report` action. */
export interface DeploymentReportPayload {
  target: TargetSpecDict;
  available: boolean;
  deployable: boolean;
  nodes: DeploymentNodes;
  constraints: Record<string, unknown>;
  validation: NirValidationPayload | null;
  /** Additive executed-substitution section; absent on older reports. */
  rewrite?: RewritePayload | null;
  notes: string[];
}

/** Payload emitted by the `deploy_run` action. */
export interface BackendRunPayload {
  target: string;
  status: BackendStatus;
  steps: number;
  path: string | null;
  readout: number[];
  spike_nodes: string[];
  membrane_nodes: string[];
  notes: string[];
  rewritten: RewritePayload | null;
  compare: BackendCompare | null;
}

/** Server message variants added by the deployment-target actions. */
export type TargetServerMsg =
  | { type: "target_list"; payload: TargetListPayload }
  | { type: "deployment_report"; payload: DeploymentReportPayload }
  | { type: "backend_run"; payload: BackendRunPayload };
