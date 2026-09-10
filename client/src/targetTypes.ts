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

/** Payload emitted by the `deployment_report` action. */
export interface DeploymentReportPayload {
  target: TargetSpecDict;
  available: boolean;
  deployable: boolean;
  nodes: DeploymentNodes;
  constraints: Record<string, unknown>;
  validation: NirValidationPayload | null;
  notes: string[];
}

/** Server message variants added by the deployment-target actions. */
export type TargetServerMsg =
  | { type: "target_list"; payload: TargetListPayload }
  | { type: "deployment_report"; payload: DeploymentReportPayload };
