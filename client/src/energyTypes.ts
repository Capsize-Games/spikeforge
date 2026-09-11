/** Protocol payloads for the event-driven energy action (Phase D3). */

/** Accounted operations: synaptic, dense MAC, and accumulate counts. */
export interface EnergyOps {
  sop: number;
  mac: number;
  ac: number;
}

/** The energy totals carried by a report; null when a target has no table. */
export interface EnergyTotals {
  sop_pj: number;
  mac_pj: number;
  ac_pj: number;
  /** Event-driven total (spikes plus neuron updates). */
  total_pj: number;
  /** Dense baseline alternative. */
  dense_pj: number;
}

/** The latency totals carried by a report; null when unavailable. */
export interface EnergyLatency {
  step_ns: number;
  total_ns: number;
}

/** The energy/latency report emitted by the `energy_report` action. */
export interface EnergyReportPayload {
  target: string;
  /** True unless a real device reported its own timing. */
  estimate: boolean;
  /** Where the numbers come from: a cost table, a device, or "unavailable". */
  basis: string;
  timesteps: number | null;
  ops: EnergyOps;
  efficiency: { sop_over_mac: number | null };
  energy: EnergyTotals | null;
  latency: EnergyLatency | null;
  measured: Record<string, unknown> | null;
  notes: string[];
}

/** Sparse-vs-dense parity and op-count comparison. */
export interface EnergyComparison {
  max_abs: number;
  tolerance: number;
  within_tolerance: boolean;
  ops: EnergyOps & { timesteps: number };
  efficiency: { sop_over_mac: number | null };
  density: number;
}

/** Payload emitted by the `energy_report` action. */
export interface EnergyPayload {
  target: string;
  report: EnergyReportPayload;
  comparison: EnergyComparison;
}

/** Server message variant added by the energy action. */
export type EnergyServerMsg = {
  type: "energy_report";
  payload: EnergyPayload;
};
