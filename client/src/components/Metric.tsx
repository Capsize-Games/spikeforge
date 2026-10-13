interface Props {
  label: string;
  value: string;
  /** The smaller in-panel size (15px) instead of the 26px metric. */
  small?: boolean;
}

/**
 * One numeric readout. The value is tabular so a changing number does not
 * shift the text beside it.
 */
export function Metric({ label, value, small = false }: Props) {
  return (
    <div className={small ? "metric sm" : "metric"}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
