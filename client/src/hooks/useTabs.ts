import { useCallback, useEffect, useRef, useState } from "react";

import { readString, STORAGE_KEYS, writeString } from "../storage";
import { DEFAULT_TAB, isTabId } from "../tabs";
import type { TabId } from "../tabs";
import { clearHighlight, findTarget } from "../tour/highlight";

/** How often a guided-tour target is re-checked while it is not rendered. */
const REVEAL_POLL_MS = 400;

/** The persisted tab, or the default when nothing valid is stored. */
function readTab(): TabId {
  const stored = readString(STORAGE_KEYS.tab, DEFAULT_TAB);
  return isTabId(stored) ? stored : DEFAULT_TAB;
}

export interface TabsState {
  active: TabId;
  select: (id: TabId) => void;
}

/**
 * Owns which tab the shell shows, persisted so a reload returns to it.
 *
 * `tourTarget` is the selector the open guided-tour step points at. An inactive
 * tab is hidden, not absent, so `document.querySelector` still finds that
 * target — the step would highlight a panel the user cannot see. Revealing the
 * tab that owns the target (via its `data-tab`) keeps the tours working.
 */
export function useTabs(tourTarget: string | null): TabsState {
  const [active, setActive] = useState<TabId>(readTab);

  // Read inside the reveal effect without making a tab change restart it:
  // switching away by hand must be respected, not immediately undone.
  const activeRef = useRef(active);
  activeRef.current = active;

  const select = useCallback((id: TabId) => {
    setActive(id);
    writeString(STORAGE_KEYS.tab, id);
  }, []);

  useEffect(() => {
    if (tourTarget === null) return;

    /** Reveal the tab owning the target; false while it has not rendered. */
    const reveal = (): boolean => {
      const element = findTarget(tourTarget);
      if (element === null) return false;
      const owner = element.closest<HTMLElement>("[data-tab]")?.dataset.tab;
      if (
        owner !== undefined &&
        isTabId(owner) &&
        owner !== activeRef.current
      ) {
        // Drop the highlight so the next tour poll re-applies it to the
        // now-visible target, which also scrolls it into view.
        clearHighlight();
        select(owner);
      }
      return true;
    };

    if (reveal()) return;
    const timer = window.setInterval(() => {
      if (reveal()) window.clearInterval(timer);
    }, REVEAL_POLL_MS);
    return () => window.clearInterval(timer);
  }, [tourTarget, select]);

  return { active, select };
}
