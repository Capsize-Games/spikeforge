/** Protocol payloads for the model-hub browser panel (Phase A4). */

import type { NirValidationPayload } from "./nirTypes";

/** Frameworks the curated catalog may declare. */
export const HUB_FRAMEWORKS: string[] = [
  "snntorch",
  "nir",
  "spikingjelly",
  "norse",
  "lava",
  "hf",
];

/** Artifact kinds a catalog entry may describe. */
export const HUB_KINDS: string[] = [
  "nir_graph",
  "state_dict",
  "framework_weights",
];

/** One availability-annotated catalog entry card. */
export interface HubEntryCard {
  id: string;
  name: string;
  framework: string;
  kind: string;
  source: string;
  license: string;
  notes: string;
  url: string | null;
  hf_repo: string | null;
  sha256: string | null;
  size_bytes: number | null;
  topology: string | null;
  input_shape: string | null;
  available: boolean;
  reason: string | null;
  cached: boolean;
}

/** Payload emitted by the `hub_list` action. */
export interface HubListPayload {
  entries: HubEntryCard[];
  /** Catalog validation problems; a malformed entry is reported here. */
  issues: string[];
}

/** Payload emitted by the `hub_search` action. */
export interface HubSearchPayload {
  query: string;
  limit: number;
  /** Whether live Hugging Face search is usable (the `hub` extra). */
  available: boolean;
  reason: string | null;
  results: HubEntryCard[];
}

/** Terminal and progress states of a hub download. */
export type HubDownloadStatus =
  | "idle"
  | "downloading"
  | "done"
  | "cancelled"
  | "error";

/** Progress streamed while the `hub_download` action runs. */
export interface HubDownloadState {
  id: string;
  status: HubDownloadStatus;
  bytes: number;
  total_bytes: number | null;
  verified: boolean;
}

/** One node in an inspected artifact's structure. */
export interface HubInspectNode {
  name: string;
  kind: string;
  params: Record<string, unknown>;
}

/** One key (or file) in an inspected artifact's structure. */
export interface HubInspectKey {
  key: string;
  shape: number[];
}

/** One directed edge in an inspected NIR graph. */
export interface HubInspectEdge {
  source: string;
  target: string;
  delayed: boolean;
}

/** Payload emitted by the `hub_inspect` action. */
export interface HubInspectPayload {
  kind: string;
  path: string;
  nodes: HubInspectNode[];
  keys: HubInspectKey[];
  edges: HubInspectEdge[];
  notes: string[];
}

/** The compatibility verdict vocabulary. */
export type HubVerdict = "exact" | "mappable" | "incompatible";

/** One named structural mismatch in an incompatible verdict. */
export interface HubMismatch {
  stage: string;
  reason: string;
  expected: string;
  actual: string;
}

/** The verdict object carried by an import result. */
export interface HubVerdictPayload {
  verdict: HubVerdict;
  topology: string | null;
  mapping: Record<string, string>;
  mismatches: HubMismatch[];
  notes: string[];
}

/** Weight-mapping report attached to an import result. */
export interface HubWeightLoad {
  loaded: boolean;
  missing: string[];
  unexpected: string[];
  mismatched: string[];
  note: string;
}

/** Payload emitted by the `hub_import` action. */
export interface HubImportPayload {
  id: string;
  kind: string;
  path: string;
  verdict: HubVerdictPayload;
  promoted: boolean;
  destination: string | null;
  meta: Record<string, unknown>;
  validation: NirValidationPayload | null;
  nir_ingestable: boolean;
  reason: string | null;
  weights: HubWeightLoad | null;
  notes: string[];
}

/** Additive filter fields sent with a hub list/search request. */
export interface HubQueryInput {
  query?: string;
  limit?: number;
  framework?: string | null;
  kind?: string | null;
  available?: boolean | null;
  topology?: string | null;
}

/** Server message variants added by the hub actions. */
export type HubServerMsg =
  | { type: "hub_list"; payload: HubListPayload }
  | { type: "hub_search"; payload: HubSearchPayload }
  | { type: "hub_download_state"; payload: HubDownloadState }
  | { type: "hub_inspect"; payload: HubInspectPayload }
  | { type: "hub_import"; payload: HubImportPayload };
