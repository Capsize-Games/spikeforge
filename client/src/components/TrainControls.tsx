import { useState } from "react";

import { TRAIN_HELP } from "../helpText";
import type { DatasetInfo, SavedModel, TrainConfig } from "../types";
import { HelpTip } from "./HelpTip";

interface Props {
  config: TrainConfig;
  datasets: DatasetInfo[];
  models: SavedModel[];
  onChange: (patch: Partial<TrainConfig>) => void;
  onTrain: () => void;
  onStop: () => void;
  onPredict: () => void;
  onSave: (name: string) => void;
  onLoad: (name: string) => void;
  onDelete: (name: string) => void;
  running: boolean;
  connected: boolean;
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  help,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  help: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">
        <span>
          {label}: <b>{value}</b>
        </span>
        <HelpTip text={help} />
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function TrainControls({
  config,
  datasets,
  models,
  onChange,
  onTrain,
  onStop,
  onPredict,
  onSave,
  onLoad,
  onDelete,
  running,
  connected,
}: Props) {
  const [saveName, setSaveName] = useState("my_model");
  const set = (patch: Partial<TrainConfig>) => onChange(patch);
  const busy = !connected || running;

  return (
    <div className="panel controls">
      <div className="panel-title">Training</div>

      <label className="field">
        <span className="field-label">
          <span>Dataset</span>
          <HelpTip text={TRAIN_HELP.dataset} />
        </span>
        <select
          value={config.dataset}
          onChange={(e) => set({ dataset: e.target.value })}
          disabled={running}
        >
          {datasets.map((d) => (
            <option key={d.name} value={d.name}>
              {d.name} ({d.classes} classes)
            </option>
          ))}
        </select>
      </label>

      <NumberField label="hidden" value={config.hidden} min={16} max={512} step={16} help={TRAIN_HELP.hidden} onChange={(v) => set({ hidden: v })} />
      <NumberField label="beta" value={config.beta} min={0.1} max={0.95} step={0.05} help={TRAIN_HELP.beta} onChange={(v) => set({ beta: v })} />
      <NumberField label="lr" value={config.lr} min={0.001} max={0.05} step={0.001} help={TRAIN_HELP.lr} onChange={(v) => set({ lr: v })} />
      <NumberField label="epochs" value={config.epochs} min={1} max={10} step={1} help={TRAIN_HELP.epochs} onChange={(v) => set({ epochs: v })} />
      <NumberField label="num_steps" value={config.num_steps} min={2} max={50} step={1} help={TRAIN_HELP.num_steps} onChange={(v) => set({ num_steps: v })} />
      <NumberField label="subset" value={config.subset} min={1} max={100} step={1} help={TRAIN_HELP.subset} onChange={(v) => set({ subset: v })} />
      <NumberField label="batch_size" value={config.batch_size} min={8} max={256} step={8} help={TRAIN_HELP.batch_size} onChange={(v) => set({ batch_size: v })} />

      <button className="apply" onClick={onTrain} disabled={busy}>
        {running ? "Training…" : "⚡ Train"}
      </button>
      <button className="apply stop" onClick={onStop} disabled={!connected || !running}>
        ■ Stop Training
      </button>
      <button className="apply ghost" onClick={onPredict} disabled={busy}>
        Predict sample
      </button>

      <div className="panel-title subsection">Model</div>
      <div className="model-row">
        <input
          className="text-input"
          value={saveName}
          onChange={(e) => setSaveName(e.target.value)}
          placeholder="checkpoint name"
        />
        <button className="apply small" onClick={() => onSave(saveName)} disabled={busy}>
          Save
        </button>
      </div>

      {models.length > 0 && (
        <div className="model-list">
          {models.map((m) => (
            <div key={m.name} className="model-item">
              <span className="model-name">{m.name}</span>
              <span className="model-actions">
                <button className="link" onClick={() => onLoad(m.name)} disabled={busy}>
                  load
                </button>
                <button className="link danger" onClick={() => onDelete(m.name)} disabled={busy}>
                  delete
                </button>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
