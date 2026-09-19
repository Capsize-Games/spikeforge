import type { TabId } from "./tabs";
import type { TranslationKey } from "./i18n/translations";

/** Interface label of each section, as it appears in the rail and header. */
export const TAB_LABEL_KEYS: Record<TabId, TranslationKey> = {
  model: "tab.model",
  viewer: "tab.viewer",
  training: "tab.training",
  hub: "tab.hub",
  deploy: "tab.deploy",
  pipeline: "tab.pipeline",
};

/** One-line description of what a section holds, used as its tooltip. */
export const TAB_HINT_KEYS: Record<TabId, TranslationKey> = {
  model: "tab.model.hint",
  viewer: "tab.viewer.hint",
  training: "tab.training.hint",
  hub: "tab.hub.hint",
  deploy: "tab.deploy.hint",
  pipeline: "tab.pipeline.hint",
};
