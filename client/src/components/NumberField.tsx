import { useEffect, useState } from "react";
import type { KeyboardEvent } from "react";

import { HelpTip } from "./HelpTip";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  /** Arrow-key and commit granularity. */
  step: number;
  help: string;
  disabled?: boolean;
  /** Value for the `data-tour` hook a guided-tour step highlights. */
  tour?: string;
  /**
   * Also render the parameter's range control beside the box. Only for the
   * settings where sweeping is how you find the value; a precise parameter
   * (a batch size, a timestep count, a learning rate) gets the box alone.
   */
  range?: boolean;
  onChange: (value: number) => void;
}

/** Digits after the point in a step: 0.001 -> 3, 8 -> 0. */
function stepDecimals(step: number): number {
  const text = String(step);
  const dot = text.indexOf(".");
  return dot < 0 ? 0 : text.length - dot - 1;
}

/** Snap onto the step grid without the float drift `0.001 * 3` adds. */
function snap(value: number, step: number): number {
  return Number((Math.round(value / step) * step).toFixed(stepDecimals(step)));
}

/** True when a number is inside the field's range. */
function inRange(value: number, min: number, max: number): boolean {
  return Number.isFinite(value) && value >= min && value <= max;
}

/**
 * The one numeric-field primitive: a labelled, keyboard-editable value with an
 * optional range control beside it.
 *
 * Typing is committed on Enter or blur; a value outside `min..max` is rejected
 * — the box falls back to the value the parameter actually holds and marks
 * itself invalid while the entry is out of range. Arrow keys step by `step`
 * and commit immediately, which is what makes this a stepper and not a text
 * box. A slider, where one is drawn, can only reach whatever the width of the
 * pane happens to represent, so it never travels without the box.
 *
 * `data-field` and `data-tour` are kept from the slider it replaces: the
 * guided tour and the end-to-end suite address parameters by them.
 */
export function NumberField({
  label,
  value,
  min,
  max,
  step,
  help,
  disabled = false,
  tour,
  range = false,
  onChange,
}: Props) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const commit = (raw: string) => {
    const parsed = Number(raw);
    if (raw.trim() === "" || !inRange(parsed, min, max)) {
      setDraft(String(value));
      return;
    }
    const next = snap(parsed, step);
    if (next !== value) onChange(next);
    setDraft(String(next));
  };

  const nudge = (direction: number) => {
    const moved = value + direction * step;
    const clamped = Math.min(max, Math.max(min, moved));
    onChange(snap(clamped, step));
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      commit(event.currentTarget.value);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      nudge(1);
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      nudge(-1);
    }
  };

  const typed = Number(draft);
  const invalid =
    draft.trim() !== "" && !inRange(typed, min, max) && !disabled;

  const box = (
    <input
      className={invalid ? "num-input invalid" : "num-input"}
      type="number"
      inputMode="decimal"
      value={draft}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      aria-label={label}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={(event) => commit(event.target.value)}
      onKeyDown={onKeyDown}
    />
  );

  return (
    <div className="field" data-field={label} data-tour={tour}>
      <span className="field-label">{label}</span>
      {range ? (
        <span className="field-scale">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            disabled={disabled}
            aria-label={label}
            onChange={(event) => onChange(Number(event.target.value))}
          />
          {box}
        </span>
      ) : (
        box
      )}
      <HelpTip text={help} />
    </div>
  );
}
