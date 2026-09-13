import { HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import type { TranslationKey } from "../i18n/translations";
import type { DatasetInfo, EncodeConfig, TrainConfig } from "../types";
import { EncodingControls } from "./EncodingControls";
import { ModelSection } from "./ModelSection";
import { SelectField } from "./SelectField";
import type { Option } from "./SelectField";
import { SliderField } from "./SliderField";
import { Section } from "./Stepper";

interface Props {
  config: EncodeConfig;
  model: TrainConfig;
  datasets: DatasetInfo[];
  gpuAvailable: boolean;
  /** Registry names for the architecture pickers, empty until listed. */
  topologies: string[];
  neurons: string[];
  surrogates: string[];
  /** True when a checkpoint is loaded: architecture/encoding are read-only. */
  locked: boolean;
  onChange: (patch: Partial<EncodeConfig>) => void;
  onModelChange: (patch: Partial<TrainConfig>) => void;
  onSelectSample: (patch: Partial<EncodeConfig>) => void;
}

/** Human label for a dataset option, including modality and availability. */
function datasetLabel(
  d: DatasetInfo,
  t: (key: TranslationKey) => string,
): string {
  const base = `${d.name} (${d.classes} ${t("dataset.classes")})`;
  if (d.modality !== "event") return base;
  if (d.available === false) {
    return `${base} — ${t("dataset.unavailable")}`;
  }
  return `${base} — ${t("dataset.events")}`;
}

/** Build the dataset dropdown, disabling event sets with no tonic loader. */
function datasetOptions(
  datasets: DatasetInfo[],
  current: string,
  t: (key: TranslationKey) => string,
): Option[] {
  if (datasets.length === 0) return [{ value: current, label: current }];
  return datasets.map((d) => ({
    value: d.name,
    label: datasetLabel(d, t),
    // Unavailable events would otherwise fail inside the download worker.
    disabled: d.modality === "event" && d.available === false,
  }));
}

export function Controls(props: Props) {
  const { t } = useI18n();
  const {
    config,
    model,
    datasets,
    gpuAvailable,
    topologies,
    neurons,
    surrogates,
    locked,
    onChange,
    onModelChange,
    onSelectSample,
  } = props;

  const selected = datasets.find((d) => d.name === config.dataset);
  const eventMode = selected?.modality === "event";
  const eventUnavailable = eventMode && selected?.available === false;

  return (
    <div className="panel controls">
      {locked && <div className="lock-note">{t("controls.locked")}</div>}

      <div className="controls-cols">
        <div className="controls-col">
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

            <SliderField
              label="subset"
              value={config.subset}
              min={1}
              max={50}
              step={1}
              help={HELP.subset}
              onChange={(v) => onChange({ subset: v })}
            />
            <SliderField
              label="batch_size"
              value={config.batch_size}
              min={8}
              max={512}
              step={8}
              help={HELP.batch_size}
              onChange={(v) => onChange({ batch_size: v })}
            />
          </Section>

          <EncodingControls
            config={config}
            locked={locked}
            eventMode={eventMode}
            onChange={onChange}
          />
        </div>

        <div className="controls-col">
          <ModelSection
            model={model}
            gpuAvailable={gpuAvailable}
            locked={locked}
            topologies={topologies}
            neurons={neurons}
            surrogates={surrogates}
            onChange={onModelChange}
          />
        </div>
      </div>
    </div>
  );
}
