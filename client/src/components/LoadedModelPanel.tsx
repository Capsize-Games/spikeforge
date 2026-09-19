import type { ModelLoadedPayload } from "../types";
import { TRAIN_HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import { FactGrid } from "./FactGrid";
import type { Fact } from "./FactGrid";
import { HelpTip } from "./HelpTip";

/**
 * Summary of the loaded checkpoint: exactly the facts the server reported
 * when it loaded, and nothing it did not.
 *
 * A field the checkpoint's record does not carry is left out rather than
 * shown as a default, because a default here would read as a measurement.
 */
export function LoadedModelPanel({ loaded }: { loaded: ModelLoadedPayload }) {
  const { t } = useI18n();
  const facts: Fact[] = [
    { label: t("field.dataset"), value: loaded.dataset },
    { label: t("loaded.accuracy"), value: `${loaded.accuracy.toFixed(1)}%` },
  ];
  if (loaded.input_mode !== undefined) {
    facts.push({ label: t("loaded.input"), value: loaded.input_mode });
  }
  if (loaded.hidden !== undefined) {
    facts.push({ label: t("loaded.hidden"), value: String(loaded.hidden) });
  }
  if (loaded.device !== undefined) {
    facts.push({ label: t("loaded.device"), value: loaded.device });
  }

  return (
    <div className="loaded-model">
      <div className="loaded-model-head">
        <span className="loaded-model-name">{loaded.name}</span>
        <HelpTip text={TRAIN_HELP.input_mode} />
      </div>
      <FactGrid facts={facts} />
    </div>
  );
}
