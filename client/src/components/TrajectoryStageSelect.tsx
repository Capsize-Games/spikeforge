interface Props {
  stages: string[];
  value: string;
  onChange: (stage: string) => void;
}

/** Compact stage picker for the trajectory viewer. */
export function TrajectoryStageSelect({ stages, value, onChange }: Props) {
  return (
    <label className="trajectory-stage">
      <span className="trajectory-stage-label">Stage</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-label="Trajectory stage"
      >
        {stages.map((stage) => (
          <option key={stage} value={stage}>
            {stage}
          </option>
        ))}
      </select>
    </label>
  );
}
