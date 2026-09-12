/**
 * The dashboard's tabs. Every panel keeps running whichever tab is focused
 * (see `TabPanel`), so this module only describes the labels and order of the
 * tab strip — it holds no state.
 */

const TAB_DEFS = [
  {
    id: "model",
    label: "Model & Data",
    hint: "Model files, dataset, encoder, and training configuration",
  },
  {
    id: "viewer",
    label: "Viewer",
    hint: "Input samples, spike activity, trajectory, and NIR graph",
  },
  {
    id: "training",
    label: "Training & Analysis",
    hint: "Live training, metrics, encoding and surrogate analysis, bench",
  },
  {
    id: "hub",
    label: "Hub",
    hint: "Search and download community models",
  },
  {
    id: "deploy",
    label: "Energy & Deployment",
    hint: "Deployment targets, capability report, and energy estimate",
  },
] as const;

/** Identifier of a tab the shell knows how to render. */
export type TabId = (typeof TAB_DEFS)[number]["id"];

export interface TabDef {
  id: TabId;
  label: string;
  hint: string;
}

/** Tab order and labels; the strip renders these and nothing else. */
export const TABS: readonly TabDef[] = TAB_DEFS;

/** The tab shown on a first visit, before any choice is persisted. */
export const DEFAULT_TAB: TabId = "viewer";

/** True when a persisted string is still one of the known tab ids. */
export function isTabId(value: string): value is TabId {
  return TAB_DEFS.some((tab) => tab.id === value);
}

/** DOM id of a tab's button, used for `aria-labelledby` on its panel. */
export function tabButtonId(id: TabId): string {
  return `tab-${id}`;
}

/** DOM id and `aria-controls` target of a tab's panel. */
export function tabPanelId(id: TabId): string {
  return `tabpanel-${id}`;
}
