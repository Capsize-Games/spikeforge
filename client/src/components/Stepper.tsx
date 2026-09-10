/** Static titled section used by the control column. */

import type { ReactNode } from "react";

export function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="section">
      <div className="section-head">
        <span className="section-text">
          <span className="section-title">{title}</span>
          {hint && <span className="section-hint">{hint}</span>}
        </span>
      </div>
      <div className="section-body">{children}</div>
    </div>
  );
}
