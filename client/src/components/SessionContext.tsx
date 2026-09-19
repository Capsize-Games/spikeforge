import { Boxes, Database } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { useI18n } from "../i18n/I18nProvider";

/** Facts the toolbar can show; each is null when it is not known yet. */
export interface SessionFacts {
  model: string | null;
  dataset: string | null;
}

interface Chip {
  key: string;
  icon: LucideIcon;
  label: string;
  value: string;
}

/**
 * The two session facts the toolbar carries: the loaded checkpoint and the
 * configured dataset.
 *
 * These are what the session is *working on*. The encoding and the device are
 * settings rather than identity, so they live in the panes that own them, and
 * the toolbar stays a header instead of a line of debug metadata. A fact whose
 * value is unknown renders nothing at all rather than a placeholder.
 */
export function SessionContext({ facts }: { facts: SessionFacts }) {
  const { t } = useI18n();

  const chips: Chip[] = [];
  if (facts.model !== null) {
    chips.push({
      key: "model",
      icon: Boxes,
      label: t("section.model"),
      value: facts.model,
    });
  }
  if (facts.dataset !== null) {
    chips.push({
      key: "dataset",
      icon: Database,
      label: t("field.dataset"),
      value: facts.dataset,
    });
  }

  if (chips.length === 0) return null;

  return (
    <div className="session-context">
      {chips.map((chip) => {
        const Icon = chip.icon;
        return (
          <span
            key={chip.key}
            className="session-chip"
            title={`${chip.label}: ${chip.value}`}
          >
            <span className="session-chip-icon">
              <Icon size={14} strokeWidth={1.75} aria-hidden="true" />
            </span>
            <span className="session-chip-value">{chip.value}</span>
          </span>
        );
      })}
    </div>
  );
}
