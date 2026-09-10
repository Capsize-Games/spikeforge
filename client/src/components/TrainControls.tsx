import { useState } from "react";

import type { SavedModel } from "../types";
import { SectionHeader } from "./Stepper";

interface Props {
  models: SavedModel[];
  running: boolean;
  connected: boolean;
  canInfer: boolean;
  onTrain: () => void;
  onStop: () => void;
  onInfer: () => void;
  onSave: (name: string) => void;
  onLoad: (name: string) => void;
  onDelete: (name: string) => void;
}

export function TrainControls({
  models,
  running,
  connected,
  canInfer,
  onTrain,
  onStop,
  onInfer,
  onSave,
  onLoad,
  onDelete,
}: Props) {
  const [saveName, setSaveName] = useState("my_model");
  const busy = !connected || running;

  return (
    <div className="panel controls">
      <SectionHeader
        step="4"
        title="Train & inspect"
        hint="fit the network, then read the results"
      />

      <button className="apply" onClick={onTrain} disabled={busy}>
        {running ? "Training…" : "⚡ Train model"}
      </button>
      <button
        className="apply stop"
        onClick={onStop}
        disabled={!connected || !running}
      >
        ■ Stop training
      </button>
      <button
        className="apply ghost"
        onClick={onInfer}
        disabled={busy || !canInfer}
      >
        🔎 Predict displayed sample
      </button>

      <div className="panel-title subsection">Saved models</div>
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
