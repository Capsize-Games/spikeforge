import { HELP } from "../helpText";
import { HelpTip } from "./HelpTip";

interface Props {
  value: [number, number] | null;
  disabled?: boolean;
  onChange: (value: [number, number] | null) => void;
}

/**
 * Parse a "HxW" entry into a sensor geometry. Returns ``null`` for a blank
 * entry (use the default), ``undefined`` for an invalid one (keep the
 * current value), and the pair otherwise.
 */
function parseSize(raw: string): [number, number] | null | undefined {
  const trimmed = raw.trim().toLowerCase();
  if (trimmed === "") return null;
  const parts = trimmed.split("x").map((part) => Number(part.trim()));
  const valid =
    parts.length === 2 && parts.every((n) => Number.isInteger(n) && n >= 4);
  return valid ? [parts[0], parts[1]] : undefined;
}

/** Optional sensor geometry override; blank keeps the default 28x28. */
export function InputSizeField({ value, disabled, onChange }: Props) {
  const text = value ? `${value[0]}x${value[1]}` : "";
  const commit = (raw: string) => {
    const parsed = parseSize(raw);
    if (parsed !== undefined) onChange(parsed);
  };
  return (
    <label className="field">
      <span className="field-label">
        input size <b>{text || "28x28"}</b>
      </span>
      <input
        key={text}
        className="text-input"
        type="text"
        placeholder="28x28"
        defaultValue={text}
        disabled={disabled}
        onBlur={(e) => commit(e.target.value)}
      />
      <HelpTip text={HELP.input_size} />
    </label>
  );
}
