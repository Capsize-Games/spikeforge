import { useState } from "react";

import type { SavedModel } from "../types";

interface Props {
  models: SavedModel[];
  current: string | null;
  connected: boolean;
  busy: boolean;
  onNew: () => void;
  onLoad: (name: string) => void;
  onSave: (name: string) => void;
}

/** Minimal model manager: start fresh, load a checkpoint, or save one. */
export function ModelPanel({
  models,
  current,
  connected,
  busy,
  onNew,
  onLoad,
  onSave,
}: Props) {
  const [selected, setSelected] = useState("");
  const [name, setName] = useState("my_model");
  const disabled = !connected || busy;

  return (
    <div className="panel model-panel">
      <div className="model-head">
        <span className="panel-title">Model</span>
        <span className={`model-chip ${current ? "on" : ""}`}>
          {current ?? "none loaded"}
        </span>
      </div>

      <div className="model-row">
        <select
          className="text-input"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={disabled || models.length === 0}
          aria-label="Saved models"
        >
          <option value="">
            {models.length ? "load saved model…" : "no saved models"}
          </option>
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
        <button
          className="apply small"
          onClick={() => selected && onLoad(selected)}
          disabled={disabled || !selected}
        >
          Load
        </button>
        <button
          className="apply small ghost"
          onClick={onNew}
          disabled={!connected}
          title="Unload the current model"
        >
          New
        </button>
      </div>

      <div className="model-row">
        <input
          className="text-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="save as…"
        />
        <button
          className="apply small"
          onClick={() => onSave(name)}
          disabled={disabled || !name.trim()}
        >
          Save
        </button>
      </div>
    </div>
  );
}
