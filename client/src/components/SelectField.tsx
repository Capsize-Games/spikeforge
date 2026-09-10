import { HelpTip } from "./HelpTip";

export interface Option {
  value: string;
  label: string;
  disabled?: boolean;
}

interface Props {
  label: string;
  value: string;
  help: string;
  options: Option[];
  disabled?: boolean;
  /** Value for the `data-tour` hook a guided-tour step highlights. */
  tour?: string;
  onChange: (value: string) => void;
}

/** A labelled select box backed by a list of options. */
export function SelectField({
  label,
  value,
  help,
  options,
  disabled,
  tour,
  onChange,
}: Props) {
  return (
    <label className="field" data-tour={tour}>
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
