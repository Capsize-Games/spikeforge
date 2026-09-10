import { HELP, TRAIN_HELP } from "../helpText";
import type { DatasetInfo, EncodeConfig, TrainConfig } from "../types";
import { CheckField, SelectField, SliderField } from "./Fields";
import { Section } from "./Stepper";
import { TrainControls } from "./TrainControls";

interface Props {
  config: EncodeConfig;
  model: TrainConfig;
  datasets: DatasetInfo[];
  gpuAvailable: boolean;
  connected: boolean;
  trainRunning: boolean;
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
    locked,
    onChange,
    onModelChange,
    onSelectSample,
    onTrain,
    onStopTrain,
  } = props;

  const set = (patch: Partial<EncodeConfig>) => onChange(patch);
  const setModel = (patch: Partial<TrainConfig>) => onModelChange(patch);

  return (
    <div className="panel controls">
      {locked && (
        <div className="lock-note">
          Locked to the loaded model so training and inference stay consistent.
          Click the ✕ on the loaded model tag in the Model panel to start fresh.
        </div>
      )}

      <Section title="Data" hint="which image the network looks at">
        <SelectField
          label="Dataset"
          value={config.dataset}
          help={HELP.dataset}
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

        <SliderField label="subset" value={config.subset} min={1} max={50} step={1} help={HELP.subset} onChange={(v) => set({ subset: v })} />
        <SliderField label="batch_size" value={config.batch_size} min={8} max={512} step={8} help={HELP.batch_size} onChange={(v) => set({ batch_size: v })} />
      </Section>

      <Section title="Encoding" hint="how the image becomes spikes">
        <SelectField
          label="Coding"
          value={config.coding}
          help={HELP.coding}
          disabled={locked}
          options={[
            { value: "rate", label: "Rate" },
            { value: "latency", label: "Latency" },
            { value: "delta", label: "Delta" },
            { value: "random", label: "Random (noise baseline)" },
          ]}
          onChange={(v) => set({ coding: v as EncodeConfig["coding"] })}
        />

        <SliderField label="num_steps" value={config.num_steps} min={5} max={200} step={5} help={HELP.num_steps} disabled={locked} onChange={(v) => set({ num_steps: v })} />
        <SliderField label="interval (ms)" value={config.interval_ms} min={20} max={400} step={10} help={HELP.interval_ms} onChange={(v) => set({ interval_ms: v })} />

        {config.coding === "rate" && (
          <SliderField label="gain" value={config.gain} min={0.05} max={1} step={0.05} help={HELP.gain} disabled={locked} onChange={(v) => set({ gain: v })} />
        )}

        {config.coding === "latency" && (
          <>
            <SliderField label="tau" value={config.tau} min={1} max={20} step={0.5} help={HELP.tau} disabled={locked} onChange={(v) => set({ tau: v })} />
            <SliderField label="threshold" value={config.threshold} min={0.005} max={0.1} step={0.005} help={HELP.threshold} disabled={locked} onChange={(v) => set({ threshold: v })} />
            <CheckField label="linear" checked={config.linear} help={HELP.linear} disabled={locked} onChange={(v) => set({ linear: v })} />
            <CheckField label="normalize" checked={config.normalize} help={HELP.normalize} disabled={locked} onChange={(v) => set({ normalize: v })} />
            <CheckField label="clip" checked={config.clip} help={HELP.clip} disabled={locked} onChange={(v) => set({ clip: v })} />
          </>
        )}

        {config.coding === "delta" && (
          <SliderField label="delta_threshold" value={config.delta_threshold} min={1} max={10} step={0.5} help={HELP.delta_threshold} disabled={locked} onChange={(v) => set({ delta_threshold: v })} />
        )}

        {config.coding === "random" && (
          <SliderField label="random_scale" value={config.random_scale} min={0.1} max={1} step={0.05} help={HELP.random_scale} disabled={locked} onChange={(v) => set({ random_scale: v })} />
        )}

      </Section>

      <Section title="Model" hint="the network that learns these spikes">
        <SelectField
          label="Device"
          value={model.device}
          help={TRAIN_HELP.device}
          options={[
            { value: "auto", label: "Auto (pick the faster)" },
            { value: "gpu", label: gpuAvailable ? "GPU" : "GPU (unavailable)", disabled: !gpuAvailable },
            { value: "cpu", label: "CPU" },
          ]}
          onChange={(v) => setModel({ device: v as TrainConfig["device"] })}
        />

        <SliderField label="hidden" value={model.hidden} min={16} max={512} step={16} help={TRAIN_HELP.hidden} disabled={locked} onChange={(v) => setModel({ hidden: v })} />
        <SliderField label="beta" value={model.beta} min={0.1} max={0.95} step={0.05} help={TRAIN_HELP.beta} disabled={locked} onChange={(v) => setModel({ beta: v })} />
        <SliderField label="lr" value={model.lr} min={0.001} max={0.05} step={0.001} help={TRAIN_HELP.lr} onChange={(v) => setModel({ lr: v })} />
        <SliderField label="epochs" value={model.epochs} min={1} max={10} step={1} help={TRAIN_HELP.epochs} onChange={(v) => setModel({ epochs: v })} />
      </Section>

      <Section title="Train & inspect" hint="fit the network, manage checkpoints">
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
