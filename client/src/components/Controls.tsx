import { HELP } from "../helpText";
import type { DatasetInfo, EncodeConfig, TrainConfig } from "../types";
import { CheckField } from "./CheckField";
import { ModelSection } from "./ModelSection";
import { SelectField } from "./SelectField";
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

  const set = (patch: Partial<EncodeConfig>) => onChange(patch);

  return (
    <div className="panel controls">
      {locked && (
        <div className="lock-note">
          Locked to the loaded model so training and inference stay
          consistent. Click the ✕ on the loaded model tag in the Model panel
          to start fresh.
        </div>
      )}

      <Section title="Data" hint="which image the network looks at">
        <SelectField
          label="Dataset"
          value={config.dataset}
          help={HELP.dataset}
          tour="dataset"
          disabled={locked}
          options={
            datasets.length === 0
              ? [{ value: config.dataset, label: config.dataset }]
              : datasets.map((d) => ({
                  value: d.name,
                  label: `${d.name} (${d.classes} classes)`,
                }))
          }
          onChange={(v) => onSelectSample({ dataset: v, sample_index: 0 })}
        />

        <SliderField
          label="subset" value={config.subset} min={1} max={50} step={1}
          help={HELP.subset} onChange={(v) => set({ subset: v })}
        />
        <SliderField
          label="batch_size" value={config.batch_size}
          min={8} max={512} step={8}
          help={HELP.batch_size} onChange={(v) => set({ batch_size: v })}
        />
      </Section>

      <Section title="Encoding" hint="how the image becomes spikes">
        <SelectField
          label="Coding"
          value={config.coding}
          help={HELP.coding}
          tour="coding"
          disabled={locked}
          options={[
            { value: "rate", label: "Rate" },
            { value: "latency", label: "Latency" },
            { value: "delta", label: "Delta" },
            { value: "random", label: "Random (noise baseline)" },
          ]}
          onChange={(v) => set({ coding: v as EncodeConfig["coding"] })}
        />

        <SliderField
          label="num_steps" value={config.num_steps}
          min={5} max={200} step={5}
          help={HELP.num_steps} disabled={locked}
          onChange={(v) => set({ num_steps: v })}
        />
        <SliderField
          label="interval (ms)" value={config.interval_ms}
          min={20} max={400} step={10}
          help={HELP.interval_ms}
          onChange={(v) => set({ interval_ms: v })}
        />

        {config.coding === "rate" && (
          <SliderField
            label="gain" value={config.gain} min={0.05} max={1} step={0.05}
            help={HELP.gain} disabled={locked}
            onChange={(v) => set({ gain: v })}
          />
        )}

        {config.coding === "latency" && (
          <>
            <SliderField
              label="tau" value={config.tau} min={1} max={20} step={0.5}
              help={HELP.tau} disabled={locked}
              onChange={(v) => set({ tau: v })}
            />
            <SliderField
              label="threshold" value={config.threshold}
              min={0.005} max={0.1} step={0.005}
              help={HELP.threshold} disabled={locked}
              onChange={(v) => set({ threshold: v })}
            />
            <CheckField
              label="linear" checked={config.linear} disabled={locked}
              help={HELP.linear} onChange={(v) => set({ linear: v })}
            />
            <CheckField
              label="normalize" checked={config.normalize} disabled={locked}
              help={HELP.normalize}
              onChange={(v) => set({ normalize: v })}
            />
            <CheckField
              label="clip" checked={config.clip} disabled={locked}
              help={HELP.clip} onChange={(v) => set({ clip: v })}
            />
          </>
        )}

        {config.coding === "delta" && (
          <SliderField
            label="delta_threshold" value={config.delta_threshold}
            min={1} max={10} step={0.5}
            help={HELP.delta_threshold} disabled={locked}
            onChange={(v) => set({ delta_threshold: v })}
          />
        )}

        {config.coding === "random" && (
          <SliderField
            label="random_scale" value={config.random_scale}
            min={0.1} max={1} step={0.05}
            help={HELP.random_scale}
            onChange={(v) => set({ random_scale: v })}
          />
        )}
      </Section>

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
