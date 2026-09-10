/** Data model for the in-app guided walkthroughs (Phase 3e). */

/** One highlighted beat inside a lesson. */
export interface TourStep {
  /** Short imperative title shown at the top of the card. */
  title: string;
  /** One or two sentences explaining what the target does and why. */
  body: string;
  /**
   * CSS selector for the element to highlight. Kept as a plain selector so a
   * step never imports a component; the attributes live in the DOM.
   */
  target: string;
  /**
   * Shown instead of the highlight when the target is not rendered (a gated
   * panel or an empty capture). Without it an absent target is silent.
   */
  absentNote?: string;
}

/** A tutorial-aligned lesson made of ordered steps. */
export interface TourLesson {
  /** Stable id used for the open/close state and React keys. */
  id: string;
  /** Name of the snnTorch tutorial this lesson follows. */
  name: string;
  /** One-line promise shown in the launcher list. */
  summary: string;
  steps: TourStep[];
}
