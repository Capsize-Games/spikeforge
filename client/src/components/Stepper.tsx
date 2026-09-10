/** Guidance chrome: the overall workflow and per-section headings. */

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

export function SectionHeader({
  step,
  title,
  hint,
}: {
  step: string;
  title: string;
  hint?: string;
}) {
  return (
    <div className="section-head">
      <span className="section-step">{step}</span>
      <span className="section-text">
        <span className="section-title">{title}</span>
        {hint && <span className="section-hint">{hint}</span>}
      </span>
    </div>
  );
}
