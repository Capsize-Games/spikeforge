import { useEffect, useRef, useState } from "react";

import { HELP, TRAIN_HELP } from "../helpText";
import type {
  DatasetInfo,
  EncodeConfig,
  SavedModel,
  TrainConfig,
} from "../types";
import { CheckField, SelectField, SliderField } from "./Fields";
import { HelpTip } from "./HelpTip";
import { Section } from "./Stepper";
import { TrainControls } from "./TrainControls";

interface Props {
  config: EncodeConfig;
  model: TrainConfig;
  datasets: DatasetInfo[];
  models: SavedModel[];
  gpuAvailable: boolean;
  connected: boolean;
  trainRunning: boolean;
  canInfer: boolean;
  onChange: (patch: Partial<EncodeConfig>) => void;
  onModelChange: (patch: Partial<TrainConfig>) => void;
  onSelectSample: (patch: Partial<EncodeConfig>) => void;
  onTrain: () => void;
  onStopTrain: () => void;
  onInfer: () => void;
  onSave: (name: string) => void;
  onLoad: (name: string) => void;
  onDelete: (name: string) => void;
}

export function Controls(props: Props) {
  const {
    config,
    model,
    datasets,
    models,
    gpuAvailable,
    connected,
    trainRunning,
    canInfer,
    onChange,
    onModelChange,
    onSelectSample,
    onTrain,
    onStopTrain,
    onInfer,
    onSave,
    onLoad,
    onDelete,
  } = props;

  const set = (patch: Partial<EncodeConfig>) => onChange(patch);
  const setModel = (patch: Partial<TrainConfig>) => onModelChange(patch);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const toggle = (id: string) =>
    setOpen((o) => ({ ...o, [id]: !o[id] }));

  const [indexText, setIndexText] = useState(String(config.sample_index));
  const debounce = useRef<number | null>(null);

  useEffect(() => {
    setIndexText(String(config.sample_index));
  }, [config.sample_index]);

  useEffect(
    () => () => {
      if (debounce.current !== null) window.clearTimeout(debounce.current);
    },
    [],
  );

  /** Debounce typed indices so each keystroke doesn't rebuild the engine. */
  const onIndexInput = (value: string) => {
    setIndexText(value);
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    if (debounce.current !== null) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      onSelectSample({ sample_index: Math.max(0, Math.round(parsed)) });
    }, 350);
  };

  return (
    <div className="panel controls">
      <Section
        step="1"
        title="Data"
        hint="which image the network looks at"
        open={!!open["1"]}
        onToggle={() => toggle("1")}
      >
        <SelectField
          label="Dataset"
          value={config.dataset}
          help={HELP.dataset}
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

        <div className="field">
          <span className="field-label">
            <span>Sample index</span>
            <HelpTip text={HELP.sample_index} />
          </span>
          <div className="sample-row">
            <button
              className="step-btn"
              onClick={() =>
                onSelectSample({
                  sample_index: Math.max(0, config.sample_index - 1),
                })
              }
              disabled={config.sample_index <= 0}
              aria-label="Previous sample"
            >
              ◀
            </button>
            <input
              className="text-input"
              type="number"
              min={0}
              value={indexText}
              onChange={(e) => onIndexInput(e.target.value)}
            />
            <button
              className="step-btn"
              onClick={() =>
                onSelectSample({ sample_index: config.sample_index + 1 })
              }
              aria-label="Next sample"
            >
              ▶
            </button>
          </div>
        </div>

        <SliderField label="subset" value={config.subset} min={1} max={50} step={1} help={HELP.subset} onChange={(v) => set({ subset: v })} />
        <SliderField label="batch_size" value={config.batch_size} min={8} max={512} step={8} help={HELP.batch_size} onChange={(v) => set({ batch_size: v })} />
      </Section>

      <Section
        step="2"
        title="Encoding"
        hint="how the image becomes spikes"
        open={!!open["2"]}
        onToggle={() => toggle("2")}
      >
        <SelectField
          label="Coding"
          value={config.coding}
          help={HELP.coding}
          options={[
            { value: "rate", label: "Rate" },
            { value: "latency", label: "Latency" },
            { value: "delta", label: "Delta" },
            { value: "random", label: "Random (noise baseline)" },
          ]}
          onChange={(v) => set({ coding: v as EncodeConfig["coding"] })}
        />

        <SliderField label="num_steps" value={config.num_steps} min={5} max={200} step={5} help={HELP.num_steps} onChange={(v) => set({ num_steps: v })} />
        <SliderField label="interval (ms)" value={config.interval_ms} min={20} max={400} step={10} help={HELP.interval_ms} onChange={(v) => set({ interval_ms: v })} />

        {config.coding === "rate" && (
          <SliderField label="gain" value={config.gain} min={0.05} max={1} step={0.05} help={HELP.gain} onChange={(v) => set({ gain: v })} />
        )}

        {config.coding === "latency" && (
          <>
            <SliderField label="tau" value={config.tau} min={1} max={20} step={0.5} help={HELP.tau} onChange={(v) => set({ tau: v })} />
            <SliderField label="threshold" value={config.threshold} min={0.005} max={0.1} step={0.005} help={HELP.threshold} onChange={(v) => set({ threshold: v })} />
            <CheckField label="linear" checked={config.linear} help={HELP.linear} onChange={(v) => set({ linear: v })} />
            <CheckField label="normalize" checked={config.normalize} help={HELP.normalize} onChange={(v) => set({ normalize: v })} />
            <CheckField label="clip" checked={config.clip} help={HELP.clip} onChange={(v) => set({ clip: v })} />
          </>
        )}

        {config.coding === "delta" && (
          <SliderField label="delta_threshold" value={config.delta_threshold} min={1} max={10} step={0.5} help={HELP.delta_threshold} onChange={(v) => set({ delta_threshold: v })} />
        )}

        {config.coding === "random" && (
          <SliderField label="random_scale" value={config.random_scale} min={0.1} max={1} step={0.05} help={HELP.random_scale} onChange={(v) => set({ random_scale: v })} />
        )}

      </Section>

      <Section
        step="3"
        title="Model"
        hint="the network that learns these spikes"
        open={!!open["3"]}
        onToggle={() => toggle("3")}
      >
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

        <SliderField label="hidden" value={model.hidden} min={16} max={512} step={16} help={TRAIN_HELP.hidden} onChange={(v) => setModel({ hidden: v })} />
        <SliderField label="beta" value={model.beta} min={0.1} max={0.95} step={0.05} help={TRAIN_HELP.beta} onChange={(v) => setModel({ beta: v })} />
        <SliderField label="lr" value={model.lr} min={0.001} max={0.05} step={0.001} help={TRAIN_HELP.lr} onChange={(v) => setModel({ lr: v })} />
        <SliderField label="epochs" value={model.epochs} min={1} max={10} step={1} help={TRAIN_HELP.epochs} onChange={(v) => setModel({ epochs: v })} />
      </Section>

      <Section
        step="4"
        title="Train & inspect"
        hint="fit the network, manage checkpoints"
        open={!!open["4"]}
        onToggle={() => toggle("4")}
      >
        <TrainControls
          models={models}
          running={trainRunning}
          connected={connected}
          canInfer={canInfer}
          onTrain={onTrain}
          onStop={onStopTrain}
          onInfer={onInfer}
          onSave={onSave}
          onLoad={onLoad}
          onDelete={onDelete}
        />
      </Section>
    </div>
  );
}
