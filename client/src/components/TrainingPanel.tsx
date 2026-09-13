import type { TrainMetrics } from "../types";
import { useI18n } from "../i18n/I18nProvider";
import { LineChart } from "./LineChart";
import { CHART_COLORS } from "./chartColors";

interface Props {
  loss: number[];
  trainAccuracy: number[];
  testAccuracy: number[];
  last: TrainMetrics | null;
  device: string | null;
}

export function TrainingPanel({
  loss,
  trainAccuracy,
  testAccuracy,
  last,
  device,
}: Props) {
  const { t } = useI18n();
  return (
    <div className="panel">
      <div className="panel-title">{t("training.title")}</div>

      <div className="panel-title subsection">{t("training.status")}</div>
      {last ? (
        <div className="metrics" data-tour="training-live">
          <div>
            step {last.step} / {last.total}
          </div>
          <div>epoch {last.epoch + 1}</div>
          <div>loss {last.loss.toFixed(3)}</div>
          <div>
            held-out{" "}
            {last.test_accuracy !== null
              ? `${last.test_accuracy.toFixed(1)}%`
              : "—"}
          </div>
          <div>batch {(last.train_accuracy * 100).toFixed(1)}%</div>
          <div>device {device ?? last.device ?? "—"}</div>
          <div>
            {last.step_ms !== undefined
              ? `${last.step_ms.toFixed(0)} ms/step`
              : "— ms/step"}
          </div>
        </div>
      ) : (
        <div className="muted">{t("training.idle")}</div>
      )}

      <div className="panel-title subsection">{t("training.loss")}</div>
      <LineChart
        bare
        series={[{ label: "loss", color: CHART_COLORS.loss, values: loss }]}
      />

      <div className="panel-title subsection">{t("training.accuracy")}</div>
      <LineChart
        bare
        series={[
          {
            label: "held-out",
            color: CHART_COLORS.heldOut,
            values: testAccuracy,
            max: 100,
          },
          {
            label: "train batch",
            color: CHART_COLORS.trainBatch,
            values: trainAccuracy,
            max: 100,
          },
        ]}
      />
    </div>
  );
}
