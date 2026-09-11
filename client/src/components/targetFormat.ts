/** Format helpers for the deployment-target and backend-run views. */

import type { RewriteRecord } from "../targetTypes";

/** Format one rewrite record as a readable, bucket-agnostic line. */
export function describeRewrite(record: RewriteRecord): string {
  if (record.primitive) {
    return `${record.node} · ${record.primitive} · ${record.reason ?? ""}`;
  }
  const detail = record.detail ?? record.reason ?? "";
  const move = `${record.from ?? "?"}→${record.to ?? "?"}`;
  return `${record.node} · ${move}${detail ? ` · ${detail}` : ""}`;
}

/** Format a single deployment constraint without hiding its value. */

/** Render a constraint value as text, keeping structured values readable. */
export function formatConstraint(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}
