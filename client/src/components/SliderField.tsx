import { HelpTip } from "./HelpTip";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  help: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}

/** A labelled range slider with a help tooltip. */
export function SliderField({
  label,
  value,
  min,
  max,
  step,
  help,
  disabled,
  onChange,
}: Props) {
  return (
    <label className="field">
      <span className="field-label">
        <span>
          {label}: <b>{value}</b>
        </span>
        <HelpTip text={help} />
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
