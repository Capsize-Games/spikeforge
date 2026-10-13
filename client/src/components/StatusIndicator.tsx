interface Props {
  /** True for a live/healthy state, false for the failure state. */
  ok: boolean;
  label: string;
  /** Extra classes, e.g. the footer's `.conn` placement hook. */
  className?: string;
  testId?: string;
}

/** A dot plus a label, colour-coded by state. */
export function StatusIndicator({ ok, label, className, testId }: Props) {
  const classes = ["status-indicator", ok ? "ok" : "bad"];
  if (className !== undefined) classes.push(className);
  return (
    <span className={classes.join(" ")} data-testid={testId}>
      <span className={ok ? "dot ok" : "dot bad"} />
      {label}
    </span>
  );
}
