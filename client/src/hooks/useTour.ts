import { useCallback, useEffect, useState } from "react";

import {
  clearHighlight,
  findTarget,
  highlightTarget,
  TOUR_TARGET_CLASS,
} from "../tour/highlight";
import type { TourLesson, TourStep } from "../tour/types";

/** How often the target is re-checked while a step is open, in ms. */
const TARGET_POLL_MS = 400;

interface TourState {
  /** Whether the launcher's lesson menu is open. */
  menuOpen: boolean;
  /** The open lesson, or null when no tour is running. */
  lesson: TourLesson | null;
  /** The current step, or null when no tour is running. */
  step: TourStep | null;
  stepIndex: number;
  /** True when the current step's target is not rendered right now. */
  missing: boolean;
  toggleMenu: () => void;
  closeMenu: () => void;
  openLesson: (id: string) => void;
  closeLesson: () => void;
  next: () => void;
  prev: () => void;
}

/**
 * Owns the walkthrough state and the highlight. The highlight lives in the
 * DOM rather than in component state so a step can point at any panel without
 * threading props through it. A poll keeps it correct: it applies the
 * highlight as soon as a gated panel or empty capture renders, and re-applies
 * it if React later rewrites the target node's className.
 */
export function useTour(lessons: TourLesson[]): TourState {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);
  const [missing, setMissing] = useState(false);

  const lesson =
    activeId === null
      ? null
      : lessons.find((item) => item.id === activeId) ?? null;
  const step = lesson === null ? null : lesson.steps[stepIndex] ?? null;
  const target = step === null ? null : step.target;

  useEffect(() => {
    if (target === null) {
      clearHighlight();
      return;
    }

    /** Highlight the target if present; otherwise flag it as absent. */
    const attempt = (): void => {
      const element = findTarget(target);
      if (element === null) {
        setMissing(true);
        return;
      }
      if (!element.classList.contains(TOUR_TARGET_CLASS)) {
        highlightTarget(element);
      }
      setMissing(false);
    };

    attempt();
    const timer = window.setInterval(attempt, TARGET_POLL_MS);
    return () => {
      window.clearInterval(timer);
      clearHighlight();
    };
  }, [target]);

  const openLesson = useCallback((id: string) => {
    setActiveId(id);
    setStepIndex(0);
    setMenuOpen(false);
  }, []);

  const closeLesson = useCallback(() => {
    setActiveId(null);
    setStepIndex(0);
  }, []);

  const toggleMenu = useCallback(() => setMenuOpen((open) => !open), []);
  const closeMenu = useCallback(() => setMenuOpen(false), []);

  const next = useCallback(() => {
    if (lesson === null) return;
    setStepIndex((index) => Math.min(index + 1, lesson.steps.length - 1));
  }, [lesson]);

  const prev = useCallback(() => {
    setStepIndex((index) => Math.max(index - 1, 0));
  }, []);

  return {
    menuOpen,
    lesson,
    step,
    stepIndex,
    missing,
    toggleMenu,
    closeMenu,
    openLesson,
    closeLesson,
    next,
    prev,
  };
}
