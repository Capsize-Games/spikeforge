/** Formatting helpers for the encoding report panel. */

import type { EncodingReportPayload } from "../introspectionTypes";

/** Flatten a [C,H,W] reconstruction to the first channel's 2D plane. */
export function firstPlane(
  reconstruction: number[][][] | null,
): number[][] | null {
  if (!reconstruction || reconstruction.length === 0) return null;
  return reconstruction[0];
}

/** Render a tensor shape as "C×H×W". */
export function formatShape(shape: number[]): string {
  return shape.length > 0 ? shape.join("×") : "—";
}

/** One key/value row in the free-form stats readout. */
export interface StatEntry {
  key: string;
  value: string;
}

/** Render a stat value; nulls and floats stay honest. */
export function formatStat(value: number | string | boolean | null): string {
  if (value === null) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? `${value}` : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  return value;
}

/** Turn the free-form stats map into stable, sorted rows. */
export function statEntries(
  stats: EncodingReportPayload["stats"],
): StatEntry[] {
  return Object.entries(stats)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => ({ key, value: formatStat(value) }));
}
