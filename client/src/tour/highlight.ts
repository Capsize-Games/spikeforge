/** DOM helpers for the walkthrough highlight (kept out of the hook). */

/** Class applied to whichever element the active tour step points at. */
export const TOUR_TARGET_CLASS = "tour-target";

/** Find the element a step selector points at, or null when it is absent. */
export function findTarget(selector: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(selector);
}

/** Remove the highlight from whatever currently carries it. */
export function clearHighlight(): void {
  const current = document.querySelector<HTMLElement>(
    `.${TOUR_TARGET_CLASS}`,
  );
  if (current !== null) {
    current.classList.remove(TOUR_TARGET_CLASS);
  }
}

/** Move the highlight to a new target and scroll it into view. */
export function highlightTarget(element: HTMLElement): void {
  clearHighlight();
  element.classList.add(TOUR_TARGET_CLASS);
  element.scrollIntoView({ behavior: "smooth", block: "center" });
}
