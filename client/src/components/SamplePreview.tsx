import { useI18n } from "../i18n/I18nProvider";
import { HeatmapCanvas } from "./HeatmapCanvas";

interface Props {
  /** Raw image sample, null while nothing has been encoded yet. */
  sample: number[][] | null;
  /** Whole-sample ON/OFF frame, for an event dataset. */
  eventFrame: number[][] | null;
  /** True for an event dataset, whose input is its own frame. */
  event: boolean;
}

/**
 * The configured input, drawn from the frame the server last sent for it.
 *
 * Nothing renders while there is no frame: an empty box would claim a sample
 * that does not exist, and the pane's own dataset facts already say which
 * dataset is configured.
 */
export function SamplePreview({ sample, eventFrame, event }: Props) {
  const { t } = useI18n();
  const frame = event ? eventFrame : sample;
  if (frame === null) return null;
  return (
    <div className="sample-preview">
      <HeatmapCanvas
        data={frame}
        palette={event ? "plasma" : "binary"}
        label={t("preview.sample")}
        width={192}
        height={192}
        fluid
      />
    </div>
  );
}
