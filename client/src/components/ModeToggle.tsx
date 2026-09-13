import { TRAIN_HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import type { ExecutionMode } from "../types";
import { HelpTip } from "./HelpTip";

interface Props {
  mode: ExecutionMode;
  onChange: (mode: ExecutionMode) => void;
}

const MODES: ExecutionMode[] = ["educational", "production"];

/** Segmented execution-mode control; the selected mode lives in the hook. */
export function ModeToggle({ mode, onChange }: Props) {
  const { t } = useI18n();
  return (
    <div className="mode-toggle" data-tour="mode-toggle">
      <span className="mode-label">
        <span>{t("mode.label")}</span>
        <HelpTip text={TRAIN_HELP.mode} />
      </span>
      <div
        className="mode-options"
        role="group"
        aria-label={t("mode.execution")}
      >
        {MODES.map((option) => {
          const active = mode === option;
          return (
            <button
              key={option}
              type="button"
              className={active ? "mode-option active" : "mode-option"}
              aria-pressed={active}
              onClick={() => onChange(option)}
            >
              {t(`mode.${option}`)}
            </button>
          );
        })}
      </div>
      <span className="mode-note">
        {mode === "educational" ? t("mode.captureOn") : t("mode.captureOff")}
      </span>
    </div>
  );
}
