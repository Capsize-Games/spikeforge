import { HelpTip } from "./HelpTip";

export interface Option {
  value: string;
  label: string;
  disabled?: boolean;
}

export function SliderField({
  label,
  value,
  min,
  max,
  step,
  help,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  help: string;
  onChange: (v: number) => void;
}) {
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
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function CheckField({
  label,
  checked,
  help,
  onChange,
}: {
  label: string;
  checked: boolean;
  help: string;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="field check">
      <span className="field-label">
        <span>
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
          />
          {label}
        </span>
        <HelpTip text={help} />
      </span>
    </label>
  );
}

export function SelectField({
  label,
  value,
  help,
  options,
  disabled,
  onChange,
}: {
  label: string;
  value: string;
  help: string;
  options: Option[];
  disabled?: boolean;
  onChange: (v: string) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">
        <span>{label}</span>
        <HelpTip text={help} />
      </span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
