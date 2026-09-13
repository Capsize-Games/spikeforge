import { useRef } from "react";
import type { KeyboardEvent } from "react";

import type { TabDef, TabId } from "../tabs";
import { tabButtonId, tabPanelId } from "../tabs";
import { useI18n } from "../i18n/I18nProvider";
import type { TranslationKey } from "../i18n/translations";

const LABEL_KEYS: Record<TabId, TranslationKey> = {
  model: "tab.model",
  viewer: "tab.viewer",
  training: "tab.training",
  hub: "tab.hub",
  deploy: "tab.deploy",
  pipeline: "tab.pipeline",
};

const HINT_KEYS: Record<TabId, TranslationKey> = {
  model: "tab.model.hint",
  viewer: "tab.viewer.hint",
  training: "tab.training.hint",
  hub: "tab.hub.hint",
  deploy: "tab.deploy.hint",
  pipeline: "tab.pipeline.hint",
};

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
  const { t } = useI18n();

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
      aria-label={t("tab.sections")}
      onKeyDown={onKeyDown}
    >
      {tabs.map((tab, index) => {
        const selected = tab.id === active;
        const running = busy[tab.id] === true;
        const hint = t(HINT_KEYS[tab.id]);
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
            title={running ? `${hint} — ${t("tab.running")}` : hint}
            onClick={() => onSelect(tab.id)}
          >
            {t(LABEL_KEYS[tab.id])}
            {running && <span className="shell-tab-dot" aria-hidden="true" />}
          </button>
        );
      })}
    </div>
  );
}
