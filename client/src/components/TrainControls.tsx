import { useState } from "react";

import { TRAIN_HELP } from "../helpText";
import type {
  Compatibility,
  EncodeConfig,
  SavedModel,
  TrainConfig,
} from "../types";
import { HelpTip } from "./HelpTip";

interface Props {
  config: TrainConfig;
  encode: EncodeConfig;
  models: SavedModel[];
  compatibility?: Compatibility | null;
  onChange: (patch: Partial<TrainConfig>) => void;
  onTrain: () => void;
  onStop: () => void;
  onInfer: () => void;
  onSave: (name: string) => void;
  onLoad: (name: string) => void;
  onDelete: (name: string) => void;
  running: boolean;
  connected: boolean;
  canInfer: boolean;
  gpuAvailable: boolean;
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

/** Read-only mirror of an encoding value owned by the LEFT controls. */
function MirrorField({
  label,
  value,
  help,
}: {
  label: string;
  value: string | number;
  help: string;
}) {
  return (
    <div className="field mirror">
      <span className="field-label">
        <span>
          {label}: <b>{value}</b>
        </span>
        <HelpTip text={help} />
      </span>
      <span className="mirror-note">set in Controls (left)</span>
    </div>
  );
}

/** Warning shown when a loaded checkpoint disagrees with the encoding. */
function MismatchBanner({ compatibility }: { compatibility: Compatibility }) {
  const legacy = compatibility.expected_input_mode === "raw";
  return (
    <div className="mismatch">
      <div className="mismatch-title">⚠ Checkpoint mismatch</div>
      {legacy ? (
        <div>Legacy raw checkpoint — the encoding controls are ignored.</div>
      ) : (
        <div>
          Trained with “{compatibility.expected_input_mode}” coding; current
          control is “{compatibility.current_coding}”.
        </div>
      )}
      {!compatibility.dataset_match && (
        <div>Dataset differs — inference is disabled until they match.</div>
      )}
      {!compatibility.num_steps_match && <div>Time-step count differs.</div>}
    </div>
  );
}

export function TrainControls({
  config,
  encode,
  models,
  compatibility,
  onChange,
  onTrain,
  onStop,
  onInfer,
  onSave,
  onLoad,
  onDelete,
  running,
  connected,
  canInfer,
  gpuAvailable,
}: Props) {
  const [saveName, setSaveName] = useState("my_model");
  const set = (patch: Partial<TrainConfig>) => onChange(patch);
  const busy = !connected || running;
  const mismatch =
    compatibility !== undefined &&
    compatibility !== null &&
    (!compatibility.coding_match ||
      !compatibility.dataset_match ||
      !compatibility.num_steps_match ||
      compatibility.expected_input_mode === "raw");

  return (
    <div className="panel controls">
      <div className="panel-title">Training</div>

      {mismatch && compatibility && (
        <MismatchBanner compatibility={compatibility} />
      )}

      {compatibility && (
        <div className="metrics">
          <div>input_mode: {compatibility.expected_input_mode}</div>
          <div>coding: {compatibility.current_coding}</div>
        </div>
      )}

      <MirrorField label="Dataset" value={encode.dataset} help={TRAIN_HELP.dataset} />

      <label className="field">
        <span className="field-label">
          <span>Device</span>
          <HelpTip text={TRAIN_HELP.device} />
        </span>
        <select
          value={config.device}
          onChange={(e) =>
            set({ device: e.target.value as TrainConfig["device"] })
          }
          disabled={busy}
        >
          <option value="auto">Auto (pick the faster)</option>
          <option value="gpu" disabled={!gpuAvailable}>
            GPU{gpuAvailable ? "" : " (unavailable)"}
          </option>
          <option value="cpu">CPU</option>
        </select>
      </label>

      <NumberField label="hidden" value={config.hidden} min={16} max={512} step={16} help={TRAIN_HELP.hidden} onChange={(v) => set({ hidden: v })} />
      <NumberField label="beta" value={config.beta} min={0.1} max={0.95} step={0.05} help={TRAIN_HELP.beta} onChange={(v) => set({ beta: v })} />
      <NumberField label="lr" value={config.lr} min={0.001} max={0.05} step={0.001} help={TRAIN_HELP.lr} onChange={(v) => set({ lr: v })} />
      <NumberField label="epochs" value={config.epochs} min={1} max={10} step={1} help={TRAIN_HELP.epochs} onChange={(v) => set({ epochs: v })} />
      <MirrorField label="num_steps" value={encode.num_steps} help={TRAIN_HELP.num_steps} />
      <NumberField label="subset" value={config.subset} min={1} max={100} step={1} help={TRAIN_HELP.subset} onChange={(v) => set({ subset: v })} />
      <NumberField label="batch_size" value={config.batch_size} min={8} max={256} step={8} help={TRAIN_HELP.batch_size} onChange={(v) => set({ batch_size: v })} />

      <button className="apply" onClick={onTrain} disabled={busy}>
        {running ? "Training…" : "⚡ Train"}
      </button>
      <button className="apply stop" onClick={onStop} disabled={!connected || !running}>
        ■ Stop Training
      </button>
      <button
        className="apply ghost"
        onClick={onInfer}
        disabled={busy || !canInfer}
        title={TRAIN_HELP.compatibility}
      >
        Infer on displayed sample
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
