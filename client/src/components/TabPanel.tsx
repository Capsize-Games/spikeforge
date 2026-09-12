import type { ReactNode } from "react";

import type { TabId } from "../tabs";
import { tabButtonId, tabPanelId } from "../tabs";

interface Props {
  id: TabId;
  active: TabId;
  children: ReactNode;
}

/**
 * One tab's content. An inactive tab is hidden with `display: none` rather
 * than unmounted, so panels that own live work — the WebSocket connection,
 * in-flight training, hub downloads, the time-cursor animation, energy runs —
 * keep running no matter which tab is focused.
 *
 * `data-tab` marks the owning tab for `useTabs`, which reveals the tab a
 * guided-tour step points at (its target is present but invisible otherwise).
 */
export function TabPanel({ id, active, children }: Props) {
  return (
    <div
      className="tab-panel"
      id={tabPanelId(id)}
      data-tab={id}
      role="tabpanel"
      aria-labelledby={tabButtonId(id)}
      hidden={id !== active}
    >
      {children}
    </div>
  );
}
