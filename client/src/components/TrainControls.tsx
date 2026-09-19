import { useI18n } from "../i18n/I18nProvider";
import { Button } from "./Button";

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
      <Button
        variant="primary"
        testId="train-start"
        onClick={onTrain}
        disabled={busy}
      >
        {readOnly
          ? t("training.disabled")
          : running
            ? t("training.running")
            : t("training.train")}
      </Button>

      {readOnly && <p className="arch-note">{t("training.demoDisabled")}</p>}

      {/* Stop is a routine action, so it stays neutral: a filled red button
          sitting disabled by default reads as chrome, not as a state. The
          filled danger tone is reserved for destructive confirmation. */}
      <Button
        testId="train-stop"
        onClick={onStop}
        disabled={!connected || !running}
      >
        {t("training.stop")}
      </Button>
    </div>
  );
}
