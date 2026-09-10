/** Compact, never-NaN number for drift metrics. */
export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  if (abs > 0 && abs < 1e-3) return value.toExponential(2);
  return value.toFixed(4);
}
