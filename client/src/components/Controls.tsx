import { HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import type { DatasetInfo, EncodeConfig } from "../types";
import { datasetFacts, datasetOptions } from "./datasetFields";
import { EncodingControls } from "./EncodingControls";
import { FactGrid } from "./FactGrid";
import { NumberField } from "./NumberField";
import { SamplePreview } from "./SamplePreview";
import { SelectField } from "./SelectField";
import { Section } from "./Stepper";

interface Props {
  config: EncodeConfig;
  datasets: DatasetInfo[];
  /** The input frame the server last sent, for the preview. */
  sample: number[][] | null;
  /** Whole-sample ON/OFF frame, for an event dataset. */
  eventFrame: number[][] | null;
  /** True when a checkpoint is loaded: architecture/encoding are read-only. */
  locked: boolean;
  onChange: (patch: Partial<EncodeConfig>) => void;
  onSelectSample: (patch: Partial<EncodeConfig>) => void;
}

/**
 * The data and encoding editor: the pane between the asset browser and the
 * network inspector.
 *
 * The content tiles into two columns once the pane is wide enough — dataset
 * and its sample on the left, encoding on the right — so a wide window gets
 * two readable columns instead of one run of full-width controls. Below that
 * width it is one column again.
 *
 * It answers three questions in the order they are asked — which dataset, what
 * the application knows about it (including a look at the frame itself), and
 * how that frame is turned into spikes. The notes at the top are the
 * configuration's own validation: a locked config, or a dataset whose learning
 * rules differ from the default.
 */
export function Controls({
  config,
  datasets,
  sample,
  eventFrame,
  locked,
  onChange,
  onSelectSample,
}: Props) {
  const { t } = useI18n();
  const selected = datasets.find((d) => d.name === config.dataset);
  const eventMode = selected?.modality === "event";
  const eventUnavailable = eventMode && selected?.available === false;

  return (
    <div className="controls">
      {locked && <div className="lock-note">{t("controls.locked")}</div>}

      <div className="controls-grid">
        <div className="controls-col">
          <SamplePreview
            sample={sample}
            eventFrame={eventFrame}
            event={eventMode}
          />

          <Section
            title={t("section.data")}
            hint={
              eventMode
                ? t("section.data.eventHint")
                : t("section.data.imageHint")
            }
          >
            <SelectField
              label={t("field.dataset")}
              value={config.dataset}
              help={HELP.dataset}
              tour="dataset"
              disabled={locked}
              options={datasetOptions(datasets, config.dataset, t)}
              onChange={(v) => onSelectSample({ dataset: v, sample_index: 0 })}
            />

            {eventUnavailable && (
              <p className="modality-note warn">{HELP.event_availability}</p>
            )}

            {eventMode && !eventUnavailable && (
              <p className="modality-note">{HELP.event_training}</p>
            )}

            <FactGrid facts={datasetFacts(selected, t)} />

            <NumberField
              label="subset"
              value={config.subset}
              min={1}
              max={50}
              step={1}
              help={HELP.subset}
              disabled={locked}
              onChange={(v) => onChange({ subset: v })}
            />
            <NumberField
              label="batch_size"
              value={config.batch_size}
              min={8}
              max={512}
              step={8}
              help={HELP.batch_size}
              disabled={locked}
              onChange={(v) => onChange({ batch_size: v })}
            />
          </Section>
        </div>

        <div className="controls-col">
          <EncodingControls
            config={config}
            locked={locked}
            eventMode={eventMode}
            onChange={onChange}
          />
        </div>
      </div>
    </div>
  );
}
