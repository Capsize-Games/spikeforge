import type { DriftMetrics, NirValidationLayer } from "../nirTypes";
import { formatMetric } from "./nirFormat";

interface Props {
  layers: Record<string, NirValidationLayer>;
}

/** One quantity's max|Δ| and agreement, or a dash when not captured. */
function cell(metrics: DriftMetrics | undefined): string {
  if (!metrics) return "—";
  const maxAbs = formatMetric(metrics.max_abs);
  return `${maxAbs} / ${formatMetric(metrics.agreement)}`;
}

/** Per-layer drift rows; failing layers are named and highlighted. */
export function NirDriftTable({ layers }: Props) {
  const rows = Object.entries(layers);
  if (rows.length === 0) {
    return <div className="panel-note">No per-layer metrics reported.</div>;
  }
  return (
    <table className="drift-table">
      <thead>
        <tr>
          <th>Layer</th>
          <th>Status</th>
          <th>Spikes max|Δ| / agree</th>
          <th>U max|Δ| / agree</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, layer]) => (
          <tr key={name} className={layer.within_tolerance ? "" : "bad"}>
            <td className="drift-layer">{name}</td>
            <td>{layer.within_tolerance ? "pass" : "fail"}</td>
            <td>{cell(layer.spikes)}</td>
            <td>{cell(layer.membrane)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
