import { HelpTip } from "./HelpTip";
import type { ReactNode } from "react";

interface Props {
  title: string;
  /** Help text for the question-mark tooltip beside the title. */
  hint?: string;
  /** Right-aligned controls for the header row. */
  actions?: ReactNode;
  /** Smaller in-panel variant, separated by a hairline. */
  subsection?: boolean;
  /** `data-tour` hook a guided-tour step highlights. */
  tour?: string;
  testId?: string;
}

/** A panel or section title, with optional help and right-aligned actions. */
export function PanelHeader({
  title,
  hint,
  actions,
  subsection = false,
  tour,
  testId,
}: Props) {
  return (
    <div
      className={subsection ? "panel-header subsection" : "panel-header"}
      data-tour={tour}
      data-testid={testId}
    >
      <span className="panel-header-title">
        {title}
        {hint !== undefined && <HelpTip text={hint} />}
      </span>
      {actions !== undefined && (
        <span className="panel-header-actions">{actions}</span>
      )}
    </div>
  );
}
