import { HELP } from "../helpText";
import type { EncodeConfig } from "../types";
import { HelpTip } from "./HelpTip";

interface Props {
  config: EncodeConfig;
  onChange: (patch: Partial<EncodeConfig>) => void;
}

function SliderField({
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

function CheckField({
  label,
  checked,
  help,
  onChange,
}: {
  label: string;
  checked: boolean;
  help: string;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="field check">
      <span className="field-label">
        <span>
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
          />
          {label}
        </span>
        <HelpTip text={help} />
      </span>
    </label>
  );
}

export function Controls({ config, onChange }: Props) {
  const set = (patch: Partial<EncodeConfig>) => onChange(patch);
  return (
    <div className="panel controls">
      <div className="panel-title">Controls</div>

      <label className="field">
        <span className="field-label">
          <span>Coding</span>
          <HelpTip text={HELP.coding} />
        </span>
        <select
          value={config.coding}
          onChange={(e) =>
            set({ coding: e.target.value as EncodeConfig["coding"] })
          }
        >
          <option value="rate">Rate</option>
          <option value="latency">Latency</option>
          <option value="delta">Delta</option>
          <option value="random">Random</option>
        </select>
      </label>

      <SliderField label="num_steps" value={config.num_steps} min={5} max={200} step={5} help={HELP.num_steps} onChange={(v) => set({ num_steps: v })} />
      <SliderField label="subset" value={config.subset} min={1} max={50} step={1} help={HELP.subset} onChange={(v) => set({ subset: v })} />
      <SliderField label="batch_size" value={config.batch_size} min={8} max={512} step={8} help={HELP.batch_size} onChange={(v) => set({ batch_size: v })} />
      <SliderField label="interval (ms)" value={config.interval_ms} min={20} max={400} step={10} help={HELP.interval_ms} onChange={(v) => set({ interval_ms: v })} />

      {config.coding === "rate" && (
        <>
          <SliderField label="gain" value={config.gain} min={0.05} max={1} step={0.05} help={HELP.gain} onChange={(v) => set({ gain: v })} />
          <SliderField label="vector_value" value={config.vector_value} min={0.1} max={1} step={0.05} help={HELP.vector_value} onChange={(v) => set({ vector_value: v })} />
        </>
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
    </div>
  );
}
