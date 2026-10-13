import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { IconButton } from "./IconButton";

interface Props {
  value: number;
  onChange: (index: number) => void;
}

/** Compact sample stepper: prev / numeric / next. */
export function SampleIndex({ value, onChange }: Props) {
  const [text, setText] = useState(String(value));
  const debounce = useRef<number | null>(null);

  useEffect(() => {
    setText(String(value));
  }, [value]);

  useEffect(
    () => () => {
      if (debounce.current !== null) window.clearTimeout(debounce.current);
    },
    [],
  );

  /** Debounce typed indices so each keystroke doesn't rebuild the engine. */
  const onInput = (raw: string) => {
    setText(raw);
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    if (debounce.current !== null) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      onChange(Math.max(0, Math.round(parsed)));
    }, 350);
  };

  return (
    <span className="sample-index" data-tour="sample-index">
      <IconButton
        icon={ChevronLeft}
        label="Previous sample"
        onClick={() => onChange(Math.max(0, value - 1))}
        disabled={value <= 0}
      />
      <input
        className="text-input index-input"
        type="number"
        min={0}
        value={text}
        onChange={(e) => onInput(e.target.value)}
        aria-label="Sample index"
      />
      <IconButton
        icon={ChevronRight}
        label="Next sample"
        onClick={() => onChange(value + 1)}
      />
    </span>
  );
}
