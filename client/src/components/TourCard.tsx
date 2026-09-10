import type { TourLesson, TourStep } from "../tour/types";

interface Props {
  lesson: TourLesson;
  step: TourStep;
  stepIndex: number;
  /** True when the current target is not rendered right now. */
  missing: boolean;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
}

/** Floating coach card: lesson name, one step, and its navigation. */
export function TourCard({
  lesson,
  step,
  stepIndex,
  missing,
  onPrev,
  onNext,
  onClose,
}: Props) {
  const total = lesson.steps.length;
  const last = stepIndex === total - 1;

  return (
    <aside className="tour-card" role="dialog" aria-label={lesson.name}>
      <div className="tour-card-head">
        <span className="tour-card-title">{lesson.name}</span>
        <button
          type="button"
          className="icon-btn tour-card-close"
          onClick={onClose}
          title="Close tour"
          aria-label="Close tour"
        >
          ✕
        </button>
      </div>

      <div className="tour-card-step">{step.title}</div>
      <p className="tour-card-body">{step.body}</p>

      {missing && (
        <p className="tour-card-note">
          {step.absentNote ??
            "This step's target is not on screen right now."}
        </p>
      )}

      <div className="tour-card-foot">
        <button
          type="button"
          className="icon-btn"
          onClick={onPrev}
          disabled={stepIndex === 0}
          title="Previous step"
          aria-label="Previous step"
        >
          ←
        </button>
        <span className="tour-card-count">
          {stepIndex + 1} / {total}
        </span>
        <button
          type="button"
          className="apply small"
          onClick={last ? onClose : onNext}
        >
          {last ? "Finish" : "Next"}
        </button>
      </div>
    </aside>
  );
}
