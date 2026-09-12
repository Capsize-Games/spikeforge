import { useRef } from "react";
import type { KeyboardEvent } from "react";

import type { TabDef, TabId } from "../tabs";
import { tabButtonId, tabPanelId } from "../tabs";

interface Props {
  tabs: readonly TabDef[];
  active: TabId;
  /** Tab ids with background work still running, marked with a status dot. */
  busy?: Readonly<Partial<Record<TabId, boolean>>>;
  onSelect: (id: TabId) => void;
}

/**
 * The tab strip below the top bar. Tabs only switch which panel is visible —
 * nothing is mounted or unmounted here, so a running training job, hub
 * download, or animation is unaffected by switching.
 */
export function TabBar({ tabs, active, busy = {}, onSelect }: Props) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);

  /** Standard tablist keys: move the selection and focus together. */
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const current = tabs.findIndex((tab) => tab.id === active);
    if (current < 0) return;

    let next: number;
    if (event.key === "ArrowLeft") {
      next = (current - 1 + tabs.length) % tabs.length;
    } else if (event.key === "ArrowRight") {
      next = (current + 1) % tabs.length;
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = tabs.length - 1;
    } else {
      return;
    }

    const tab = tabs[next];
    if (tab === undefined) return;
    event.preventDefault();
    onSelect(tab.id);
    buttons.current[next]?.focus();
  };

  return (
    <div
      className="tabbar"
      role="tablist"
      aria-label="Dashboard sections"
      onKeyDown={onKeyDown}
    >
      {tabs.map((tab, index) => {
        const selected = tab.id === active;
        const running = busy[tab.id] === true;
        return (
          <button
            key={tab.id}
            ref={(element) => {
              buttons.current[index] = element;
            }}
            type="button"
            id={tabButtonId(tab.id)}
            className={selected ? "shell-tab active" : "shell-tab"}
            role="tab"
            aria-selected={selected}
            aria-controls={tabPanelId(tab.id)}
            tabIndex={selected ? 0 : -1}
            title={running ? `${tab.hint} — running` : tab.hint}
            onClick={() => onSelect(tab.id)}
          >
            {tab.label}
            {running && <span className="shell-tab-dot" aria-hidden="true" />}
          </button>
        );
      })}
    </div>
  );
}
