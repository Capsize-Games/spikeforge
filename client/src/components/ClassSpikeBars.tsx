import type { InferencePayload } from "../types";
import { LineChart } from "./LineChart";

interface Props {
  classSpikes: number[];
  predicted?: number | null;
  trueLabel?: number | null;
  outputOverTime?: number[][];
}

const PALETTE = [
  "#58a6ff",
  "#f85149",
  "#3fb950",
  "#e3b341",
  "#a371f7",
  "#39c5cf",
  "#ff7b72",
  "#7ee787",
  "#d2a8ff",
  "#ffa657",
];

/** One series per output class so the chart shows the spike race. */
function toSeries(matrix: number[][]) {
  const classes = matrix[0]?.length ?? 0;
  const max = Math.max(1e-6, ...matrix.flat());
  return Array.from({ length: classes }, (_, c) => ({
    label: String(c),
    color: PALETTE[c % PALETTE.length],
    values: matrix.map((row) => row[c] ?? 0),
    max,
  }));
}

/** Per-class output spike totals as bars, plus an over-time line chart. */
export function ClassSpikeBars({
  classSpikes,
  predicted = null,
  trueLabel = null,
  outputOverTime,
}: Props) {
  const max = Math.max(1e-6, ...classSpikes);
  const series = outputOverTime && outputOverTime.length > 1
    ? toSeries(outputOverTime)
    : null;

  return (
    <div className="class-spike-wrap">
      <div className="class-bars">
        {classSpikes.map((value, i) => {
          const cls =
            "class-bar" +
            (i === predicted ? " pred" : "") +
            (i === trueLabel ? " true" : "");
          return (
            <div key={i} className={cls} title={`class ${i}: ${value}`}>
              <div
                className="class-bar-fill"
                style={{ height: `${(value / max) * 100}%` }}
              />
              <span className="class-bar-label">{i}</span>
            </div>
          );
        })}
      </div>
      {series && (
        <LineChart
          title="Output spikes over time"
          series={series}
          width={320}
          height={130}
        />
      )}
    </div>
  );
}

/** Convenience: build the component straight from an inference payload. */
export function ClassSpikeBarsFromInference({
  inference,
}: {
  inference: InferencePayload;
}) {
  return (
    <ClassSpikeBars
      classSpikes={inference.class_spikes}
      predicted={inference.predicted}
      trueLabel={inference.true_label}
      outputOverTime={inference.output_over_time}
    />
  );
}
