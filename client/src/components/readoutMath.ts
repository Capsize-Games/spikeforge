/**
 * Client-side readout math shared by the output bars and the confidence
 * displays. These mirror the server's inference exactly: the network's logits
 * are the cumulative output spikes divided by the total step count, and the
 * reported confidence is the max softmax of those logits.
 */

/** Sum a `[T][classes]` spike matrix through and including `step`. */
export function cumulativeTotals(matrix: number[][], step: number): number[] {
  const classes = matrix[0]?.length ?? 0;
  const totals = new Array(classes).fill(0);
  const last = Math.max(0, Math.min(step, matrix.length - 1));
  for (let t = 0; t <= last; t++) {
    const row = matrix[t] ?? [];
    for (let c = 0; c < classes; c++) totals[c] += row[c] ?? 0;
  }
  return totals;
}

/** Numerically stable softmax, matching torch.softmax. */
function softmax(values: number[]): number[] {
  if (values.length === 0) return [];
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

/**
 * Per-step softmax confidence (%). Logits at step `t` are the spikes
 * accumulated over `0..t` divided by the *total* step count, so the last
 * value reproduces the server's final confidence exactly.
 */
export function confidenceSeries(matrix: number[][]): number[] {
  const steps = Math.max(1, matrix.length);
  const classes = matrix[0]?.length ?? 0;
  const cumulative = new Array(classes).fill(0);
  const series: number[] = [];
  for (let t = 0; t < matrix.length; t++) {
    const row = matrix[t] ?? [];
    for (let c = 0; c < classes; c++) cumulative[c] += (row[c] ?? 0) / steps;
    series.push(Math.max(...softmax(cumulative)) * 100);
  }
  return series;
}

/** Confidence (%) at a given step, or null when no trace is available. */
export function confidenceAt(
  matrix: number[][] | undefined,
  step: number,
): number | null {
  if (!matrix || matrix.length === 0) return null;
  const idx = Math.max(0, Math.min(step, matrix.length - 1));
  const series = confidenceSeries(matrix);
  return series[idx] ?? null;
}
