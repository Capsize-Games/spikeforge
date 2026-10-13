import { useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import {
  Activity,
  Boxes,
  BrainCircuit,
  ChartLine,
  Library,
  Moon,
  PanelLeft,
  Sun,
  Workflow,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { TAB_HINT_KEYS, TAB_LABEL_KEYS } from "../tabLabels";
import { tabButtonId, tabPanelId } from "../tabs";
import type { TabDef, TabId } from "../tabs";
import { useTheme } from "../theme";
import { useI18n } from "../i18n/I18nProvider";
import { IconButton } from "./IconButton";

/** One icon per section; the label and order come from `src/tabs.ts`. */
const ICONS: Record<TabId, LucideIcon> = {
  model: Boxes,
  viewer: Activity,
  training: ChartLine,
  hub: Library,
  deploy: Zap,
  pipeline: Workflow,
};

/**
 * The rail reads as two groups: the workspaces that build a model (Model &
 * Data, Viewer, Training) and the ones that fetch, ship or chain one (Hub,
 * Deployment, Pipeline). The break is spacing rather than a rule, so grouping
 * costs the window no extra line.
 */
const GROUP_BREAK_BEFORE: TabId = "hub";

interface Props {
  tabs: readonly TabDef[];
  active: TabId;
  /** Sections with background work still running, marked with a dot. */
  busy?: Readonly<Partial<Record<TabId, boolean>>>;
  onSelect: (id: TabId) => void;
}

/**
 * The left navigation rail. It only switches which workspace is visible —
 * nothing is mounted or unmounted here, so a running training job, hub
 * download, or animation is unaffected by switching.
 *
 * Collapsed it is 54px of 18px icons in 42px rows, with the section hint as
 * the tooltip; expanded it is 208px with labels. The theme control sits at its
 * foot with the shell's own settings. It is still the `tablist` the guided
 * tour and the end-to-end suite address (`#tab-<id>` / `#tabpanel-<id>`).
 */
export function NavRail({ tabs, active, busy = {}, onSelect }: Props) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const [expanded, setExpanded] = useState(false);
  const { t } = useI18n();
  const { theme, toggle } = useTheme();

  /** Standard tablist keys: move the selection and focus together. Up/Down
   * suit a vertical rail; Left/Right stay for muscle memory. */
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const current = tabs.findIndex((tab) => tab.id === active);
    if (current < 0) return;

    let next: number;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      next = (current + 1) % tabs.length;
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      next = (current - 1 + tabs.length) % tabs.length;
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
    <nav
      className={expanded ? "nav-rail expanded" : "nav-rail"}
      aria-label={t("tab.sections")}
    >
      {/* The product mark occupies the header's own height, so the navigation
          rows begin level with the workspace below the toolbar. */}
      <div className="nav-rail-mark" aria-hidden="true">
        <BrainCircuit size={28} strokeWidth={1.5} />
      </div>

      <div
        className="nav-rail-list"
        role="tablist"
        aria-label={t("tab.sections")}
        onKeyDown={onKeyDown}
      >
        {tabs.map((tab, index) => {
          const selected = tab.id === active;
          const running = busy[tab.id] === true;
          const Icon = ICONS[tab.id];
          const label = t(TAB_LABEL_KEYS[tab.id]);
          const hint = t(TAB_HINT_KEYS[tab.id]);
          const group = tab.id === GROUP_BREAK_BEFORE ? " group-start" : "";
          return (
            <button
              key={tab.id}
              ref={(element) => {
                buttons.current[index] = element;
              }}
              type="button"
              id={tabButtonId(tab.id)}
              className={`rail-item${selected ? " active" : ""}${group}`}
              role="tab"
              aria-selected={selected}
              aria-controls={tabPanelId(tab.id)}
              aria-label={label}
              tabIndex={selected ? 0 : -1}
              title={running ? `${hint} — ${t("tab.running")}` : hint}
              onClick={() => onSelect(tab.id)}
            >
              <span className="rail-item-icon">
                <Icon size={18} strokeWidth={1.75} aria-hidden="true" />
              </span>
              <span className="rail-item-label">{label}</span>
              {running && <span className="shell-tab-dot" aria-hidden="true" />}
            </button>
          );
        })}
      </div>

      <div className="nav-rail-foot">
        <IconButton
          icon={theme === "dark" ? Sun : Moon}
          label={t("theme.toggle")}
          title={theme === "dark" ? t("theme.light") : t("theme.dark")}
          onClick={toggle}
        />
        <IconButton
          icon={PanelLeft}
          label={expanded ? t("nav.collapse") : t("nav.expand")}
          onClick={() => setExpanded((open) => !open)}
        />
      </div>
    </nav>
  );
}
