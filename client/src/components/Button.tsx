import type { ReactNode } from "react";

/**
 * `primary` is the single filled-accent action; `danger` is destructive;
 * `ghost` is routine and borderless. The default is a neutral control.
 */
export type ButtonVariant = "neutral" | "primary" | "danger" | "ghost";

interface Props {
  children: ReactNode;
  onClick: () => void;
  variant?: ButtonVariant;
  disabled?: boolean;
  /** Toolbar-height variant. */
  small?: boolean;
  block?: boolean;
  title?: string;
  testId?: string;
}

/** Base button. Geometry and tones come from the design tokens. */
export function Button({
  children,
  onClick,
  variant = "neutral",
  disabled = false,
  small = false,
  block = false,
  title,
  testId,
}: Props) {
  const classes = ["btn"];
  if (variant !== "neutral") classes.push(variant);
  if (small) classes.push("sm");
  if (block) classes.push("block");
  return (
    <button
      type="button"
      className={classes.join(" ")}
      onClick={onClick}
      disabled={disabled}
      title={title}
      data-testid={testId}
    >
      {children}
    </button>
  );
}
