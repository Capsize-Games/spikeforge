import type { StageTrace } from "../introspectionTypes";
import { useTheme } from "../theme";
import { LineChart } from "./LineChart";
import { stageSeries } from "./trajectoryMath";

interface Props {
  trace: StageTrace | null;
  cursorIndex: number | null;
  emptyNote: string;
}

/** The U[t]/I[t] line pair for one stage of a trajectory payload. */
export function TrajectoryChart({ trace, cursorIndex, emptyNote }: Props) {
  const { colors } = useTheme();
  if (!trace) {
    return <div className="panel-note">{emptyNote}</div>;
  }
  const series = stageSeries(trace, colors.accent, colors.dot);
  return (
    <div className="trajectory-chart">
      <LineChart
        series={series}
        width={360}
        height={150}
        cursorIndex={cursorIndex}
        bare
      />
    </div>
  );
}
