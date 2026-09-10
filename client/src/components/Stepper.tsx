/** Collapsible numbered section used by the control column. */

import type { ReactNode } from "react";

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
