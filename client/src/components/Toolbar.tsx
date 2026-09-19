import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Accessible name for the toolbar. */
  label?: string;
  testId?: string;
  className?: string;
}

/** A horizontal strip of controls on a raised surface. */
export function Toolbar({ children, label, testId, className }: Props) {
  return (
    <div
      className={className === undefined ? "toolbar" : `toolbar ${className}`}
      role="toolbar"
      aria-label={label}
      data-testid={testId}
    >
      {children}
    </div>
  );
}
