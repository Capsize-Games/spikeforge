import { useState } from "react";

import type { SavedModel } from "../types";

interface Props {
  models: SavedModel[];
  current: string | null;
  connected: boolean;
  busy: boolean;
  loading: boolean;
  onNew: () => void;
  onLoad: (name: string) => void;
  onSave: (name: string) => void;
}

/**
 * Model manager with two isolated workflows: loading an existing checkpoint
 * and saving the current one. Only the primary action of the active tab is
 * highlighted, so the actions never compete for attention.
 */
export function ModelPanel({
  models,
  current,
  connected,
  busy,
  loading,
  onNew,
  onLoad,
  onSave,
}: Props) {
  const [tab, setTab] = useState<"load" | "save" | "bundle">("load");
  const [selected, setSelected] = useState("");
  const [name, setName] = useState("my_model");
  const [bundleTarget, setBundleTarget] = useState("");
  const disabled = !connected || busy || loading;

  return (
    <section className="model-panel">
      <header className="model-head">
        <span className="model-title">Model</span>
        <span className="model-current">
          <span className="model-current-label">Current:</span>
          {loading ? (
            <span className="model-loading">
              <span className="spinner" />
              loading…
            </span>
          ) : (
            <span className={`model-chip ${current ? "on" : ""}`}>
              <span className="model-chip-name">{current ?? "none"}</span>
              {current && (
                <button
                  type="button"
                  className="model-chip-x"
                  onClick={onNew}
                  disabled={!connected || loading}
                  title="Unload model"
                  aria-label="Unload model"
                >
                  ✕
                </button>
              )}
            </span>
          )}
        </span>
      </header>

      <div className="tabs" role="tablist" aria-label="Model actions">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "load"}
          className={`tab ${tab === "load" ? "active" : ""}`}
          onClick={() => setTab("load")}
        >
          Load model
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "save"}
          className={`tab ${tab === "save" ? "active" : ""}`}
          onClick={() => setTab("save")}
        >
          Save as
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "bundle"}
          className={`tab ${tab === "bundle" ? "active" : ""}`}
          onClick={() => setTab("bundle")}
        >
          Bundle
        </button>
      </div>

      {tab === "load" && (
        <div className="model-block">
          <div className="control-row">
            <select
              className="text-input"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              disabled={disabled || models.length === 0}
              aria-label="Saved models"
            >
              <option value="">
                {models.length ? "choose a saved model…" : "no saved models"}
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
          </div>
        </div>
      )}

      {tab === "save" && (
        <div className="model-block">
          <div className="control-row">
            <input
              className="text-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="model name…"
              aria-label="Model name"
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
      )}

      {tab === "bundle" && (
        <div className="model-block">
          <div className="control-row">
            <select
              className="text-input"
              value={bundleTarget}
              onChange={(e) => setBundleTarget(e.target.value)}
              disabled={models.length === 0}
              aria-label="Model to bundle"
            >
              <option value="">
                {models.length ? "choose a saved model…" : "no saved models"}
              </option>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
            <a
              className={`apply small ${bundleTarget ? "" : "disabled"}`}
              href={
                bundleTarget
                  ? `/api/bundle/${encodeURIComponent(bundleTarget)}`
                  : undefined
              }
              aria-disabled={!bundleTarget}
              onClick={(e) => {
                if (!bundleTarget) e.preventDefault();
              }}
              download
            >
              Download .spkf
            </a>
          </div>
          <p className="arch-note">
            A portable deployment bundle — everything a runtime needs to load
            and run this model, with no training code attached. Install it as
            a standalone command with{" "}
            <code>spikeforge-serve install &lt;file&gt;.spkf</code>, or serve
            it with <code>spikeforge-serve serve --bundle &lt;file&gt;.spkf</code>.
          </p>
        </div>
      )}
    </section>
  );
}
