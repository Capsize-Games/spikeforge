interface Props {
  /** Bin edges, length = counts + 1. */
  edges: number[];
  /** Per-bin counts. */
  counts: number[];
}

/** Format a bin edge compactly, tolerating a missing boundary. */
function edge(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(2);
}

/** A compact firing-rate histogram drawn as proportional bars. */
export function HistogramChart({ edges, counts }: Props) {
  const max = counts.length > 0 ? Math.max(...counts, 1) : 1;
  return (
    <div className="histogram">
      <div className="histogram-bars">
        {counts.map((count, index) => (
          <span
            key={index}
            className="histogram-bar"
            style={{ height: `${(count / max) * 100}%` }}
            title={`${edge(edges[index])}–${edge(edges[index + 1])}: ${count}`}
          />
        ))}
      </div>
      <div className="histogram-axis">
        <span>{edge(edges[0])}</span>
        <span>firing rate</span>
        <span>{edge(edges[edges.length - 1])}</span>
      </div>
    </div>
  );
}
