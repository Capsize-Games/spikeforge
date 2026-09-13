interface Props {
  running: boolean;
  connected: boolean;
  readOnly?: boolean;
  onTrain: () => void;
  onStop: () => void;
}

/** Train & inspect section content: training actions on a single row. */
export function TrainControls({
  running,
  connected,
  readOnly = false,
  onTrain,
  onStop,
}: Props) {
  const { t } = useI18n();
  const busy = !connected || running || readOnly;

  return (
    <div className="actions" data-tour="train-controls">
      <button className="apply" onClick={onTrain} disabled={busy}>
        {readOnly
          ? t("training.disabled")
          : running
            ? t("training.running")
            : t("training.train")}
      </button>
      {readOnly && <p className="arch-note">{t("training.demoDisabled")}</p>}
      <button
        className="apply stop"
        onClick={onStop}
        disabled={!connected || !running}
      >
        {t("training.stop")}
      </button>
    </div>
  );
}
import { useI18n } from "../i18n/I18nProvider";
