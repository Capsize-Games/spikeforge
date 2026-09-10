import { TRAIN_HELP } from "../helpText";
import type { ExecutionMode } from "../types";
import { HelpTip } from "./HelpTip";

interface Props {
  mode: ExecutionMode;
  onChange: (mode: ExecutionMode) => void;
}

const MODES: { value: ExecutionMode; label: string }[] = [
  { value: "educational", label: "Educational" },
  { value: "production", label: "Production" },
];

/** Segmented execution-mode control; the selected mode lives in the hook. */
export function ModeToggle({ mode, onChange }: Props) {
  return (
    <div className="mode-toggle" data-tour="mode-toggle">
      <span className="mode-label">
        <span>Mode</span>
        <HelpTip text={TRAIN_HELP.mode} />
      </span>
      <div className="mode-options" role="group" aria-label="Execution mode">
        {MODES.map((option) => {
          const active = mode === option.value;
          return (
            <button
              key={option.value}
              type="button"
              className={active ? "mode-option active" : "mode-option"}
              aria-pressed={active}
              onClick={() => onChange(option.value)}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      <span className="mode-note">
        {mode === "educational" ? "capture on" : "capture off"}
      </span>
    </div>
  );
}
