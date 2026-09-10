import type { StageTrace } from "../introspectionTypes";

/** One line handed to `LineChart` (mirrors its structural `Series`). */
export interface TraceSeries {
  label: string;
  color: string;
  values: number[];
  max: number;
}

/** Mean across the traced neurons for each step of a `[T][N]` trace. */
export function meanByStep(rows: number[][]): number[] {
  return rows.map((row) =>
    row.length === 0 ? 0 : row.reduce((sum, v) => sum + v, 0) / row.length,
  );
}

/** Largest magnitude across the traces, so both lines share one scale. */
export function sharedMax(...traces: number[][]): number {
  let max = 1e-6;
  for (const values of traces) {
    for (const value of values) max = Math.max(max, Math.abs(value));
  }
  return max;
}

/** Build the U[t] and I[t] series for one stage's bounded trace. */
export function stageSeries(
  trace: StageTrace,
  membraneColor: string,
  currentColor: string,
): TraceSeries[] {
  const membrane = meanByStep(trace.membrane);
  const current = meanByStep(trace.current);
  const max = sharedMax(membrane, current);
  const suffix = trace.neurons > 1 ? ` (mean of ${trace.neurons})` : "";
  return [
    {
      label: `U[t]${suffix}`,
      color: membraneColor,
      values: membrane,
      max,
    },
    {
      label: `I[t]${suffix}`,
      color: currentColor,
      values: current,
      max,
    },
  ];
}
