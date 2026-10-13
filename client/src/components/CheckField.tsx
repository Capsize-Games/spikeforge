import { HelpTip } from "./HelpTip";

interface Props {
  label: string;
  checked: boolean;
  help: string;
  disabled?: boolean;
  onChange: (value: boolean) => void;
}

/** A labelled checkbox with a help tooltip. */
export function CheckField({
  label,
  checked,
  help,
  disabled,
  onChange,
}: Props) {
  return (
    <label className="field check">
      <span className="field-label">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        {label}
      </span>
      <HelpTip text={help} />
    </label>
  );
}
