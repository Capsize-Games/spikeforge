import { HELP } from "../helpText";
import type { DatasetInfo, EncodeConfig, TrainConfig } from "../types";
import { EncodingControls } from "./EncodingControls";
import { ModelSection } from "./ModelSection";
import { SelectField } from "./SelectField";
import type { Option } from "./SelectField";
import { SliderField } from "./SliderField";
import { Section } from "./Stepper";
import { TrainControls } from "./TrainControls";

interface Props {
  config: EncodeConfig;
  model: TrainConfig;
  datasets: DatasetInfo[];
  gpuAvailable: boolean;
  connected: boolean;
  trainRunning: boolean;
  /** Registry names for the architecture pickers, empty until listed. */
  topologies: string[];
  neurons: string[];
  surrogates: string[];
  /** True when a checkpoint is loaded: architecture/encoding are read-only. */
  locked: boolean;
  onChange: (patch: Partial<EncodeConfig>) => void;
  onModelChange: (patch: Partial<TrainConfig>) => void;
  onSelectSample: (patch: Partial<EncodeConfig>) => void;
  onTrain: () => void;
  onStopTrain: () => void;
}

/** Human label for a dataset option, including modality and availability. */
function datasetLabel(d: DatasetInfo): string {
  const base = `${d.name} (${d.classes} classes)`;
  if (d.modality !== "event") return base;
  if (d.available === false) {
    return `${base} — unavailable (install the events extra)`;
  }
  return `${base} — events`;
}

/** Build the dataset dropdown, disabling event sets with no tonic loader. */
function datasetOptions(datasets: DatasetInfo[], current: string): Option[] {
  if (datasets.length === 0) return [{ value: current, label: current }];
  return datasets.map((d) => ({
    value: d.name,
    label: datasetLabel(d),
    // Unavailable events would otherwise fail inside the download worker.
    disabled: d.modality === "event" && d.available === false,
  }));
}

export function Controls(props: Props) {
  const {
    config,
    model,
    datasets,
    gpuAvailable,
    connected,
    trainRunning,
    topologies,
    neurons,
    surrogates,
    locked,
    onChange,
    onModelChange,
    onSelectSample,
    onTrain,
    onStopTrain,
  } = props;

  const selected = datasets.find((d) => d.name === config.dataset);
  const eventMode = selected?.modality === "event";
  const eventUnavailable = eventMode && selected?.available === false;

  return (
    <div className="panel controls">
      {locked && (
        <div className="lock-note">
          Locked to the loaded model so training and inference stay
          consistent. Click the ✕ on the loaded model tag in the Model panel
          to start fresh.
        </div>
      )}

      <Section
        title="Data"
        hint={
          eventMode
            ? "which event recording the network looks at"
            : "which image the network looks at"
        }
      >
        <SelectField
          label="Dataset"
          value={config.dataset}
          help={HELP.dataset}
          tour="dataset"
          disabled={locked}
          options={datasetOptions(datasets, config.dataset)}
          onChange={(v) => onSelectSample({ dataset: v, sample_index: 0 })}
        />

        {eventUnavailable && (
          <p className="modality-note warn">{HELP.event_availability}</p>
        )}

        <SliderField
          label="subset" value={config.subset} min={1} max={50} step={1}
          help={HELP.subset} onChange={(v) => onChange({ subset: v })}
        />
        <SliderField
          label="batch_size" value={config.batch_size}
          min={8} max={512} step={8}
          help={HELP.batch_size} onChange={(v) => onChange({ batch_size: v })}
        />
      </Section>

      <EncodingControls
        config={config}
        locked={locked}
        eventMode={eventMode}
        onChange={onChange}
      />

      <ModelSection
        model={model}
        gpuAvailable={gpuAvailable}
        locked={locked}
        topologies={topologies}
        neurons={neurons}
        surrogates={surrogates}
        onChange={onModelChange}
      />

      <Section
        title="Train & inspect"
        hint="fit the network, manage checkpoints"
      >
        <TrainControls
          running={trainRunning}
          connected={connected}
          onTrain={onTrain}
          onStop={onStopTrain}
        />
      </Section>
    </div>
  );
}
