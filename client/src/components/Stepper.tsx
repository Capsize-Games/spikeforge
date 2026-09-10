/** Guidance chrome: the overall workflow and collapsible sections. */

import type { ReactNode } from "react";

export function Stepper({
  steps,
}: {
  steps: { n: string; label: string }[];
}) {
  return (
    <nav className="stepper" aria-label="Workflow">
      {steps.map((s, i) => (
        <span key={s.n} className="stepper-item">
          <b>{s.n}</b> {s.label}
          {i < steps.length - 1 && <span className="stepper-arrow">→</span>}
        </span>
      ))}
    </nav>
  );
}

export function Section({
  step,
  title,
  hint,
  open,
  onToggle,
  children,
}: {
  step: string;
  title: string;
  hint?: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <div className={`section ${open ? "open" : ""}`}>
      <button
        type="button"
        className="section-head"
        aria-expanded={open}
        onClick={onToggle}
      >
        <span className="section-step">{step}</span>
        <span className="section-text">
          <span className="section-title">{title}</span>
          {hint && <span className="section-hint">{hint}</span>}
        </span>
        <span className="section-caret">{open ? "▾" : "▸"}</span>
      </button>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
}
