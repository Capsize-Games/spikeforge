import { useState } from "react";

import { accessToken } from "../accessToken";
import { useI18n } from "../i18n/I18nProvider";
import { ConfirmDialog } from "./ConfirmDialog";
import type { SavedModel } from "../types";

/** `/api/bundle/<name>`, carrying the access token when the page has one. */
function bundleHref(name: string): string {
  const token = accessToken();
  const query = token ? `?token=${encodeURIComponent(token)}` : "";
  return `/api/bundle/${encodeURIComponent(name)}${query}`;
}

interface Props {
  models: SavedModel[];
  current: string | null;
  connected: boolean;
  busy: boolean;
  loading: boolean;
  onNew: () => void;
  onLoad: (name: string) => void;
  onSave: (name: string) => void;
  readOnly?: boolean;
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
  readOnly = false,
}: Props) {
  const { t } = useI18n();
  const [tab, setTab] = useState<"load" | "save" | "bundle">("load");
  const [selected, setSelected] = useState("");
  const [name, setName] = useState("my_model");
  const [bundleTarget, setBundleTarget] = useState("");
  const [confirmUnload, setConfirmUnload] = useState(false);
  const disabled = !connected || busy || loading || readOnly;

  return (
    <section className="model-panel" data-testid="model-panel">
      {/* The pane heading already names this pane, so the row starts with the
          state that changes: which checkpoint is loaded. */}
      <header className="model-head">
        <span className="model-current">
          <span className="model-current-label">{t("model.current")}</span>
          {loading ? (
            <span className="model-loading">
              <span className="spinner" />
              {t("model.loading")}
            </span>
          ) : (
            <span className={`model-chip ${current ? "on" : ""}`}>
              <span className="model-chip-name" data-testid="model-current">
                {current ?? t("model.none")}
              </span>
              {current && (
                <button
                  type="button"
                  className="model-chip-x"
                  onClick={() => setConfirmUnload(true)}
                  disabled={!connected || loading || readOnly}
                  title={t("model.unload")}
                  aria-label={t("model.unload")}
                >
                  ✕
                </button>
              )}
            </span>
          )}
        </span>
      </header>

      <div className="tabs" role="tablist" aria-label={t("model.actions")}>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "load"}
          data-testid="model-tab-load"
          className={`tab ${tab === "load" ? "active" : ""}`}
          onClick={() => setTab("load")}
          disabled={readOnly}
        >
          {t("model.load")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "save"}
          data-testid="model-tab-save"
          className={`tab ${tab === "save" ? "active" : ""}`}
          onClick={() => setTab("save")}
          disabled={readOnly}
        >
          {t("model.saveAs")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "bundle"}
          data-testid="model-tab-bundle"
          className={`tab ${tab === "bundle" ? "active" : ""}`}
          onClick={() => setTab("bundle")}
        >
          {t("model.bundle")}
        </button>
      </div>

      {tab === "load" && (
        <div className="model-block">
          {readOnly && <p className="arch-note">{t("model.demoDisabled")}</p>}
          <div className="control-row">
            <select
              className="text-input"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              disabled={disabled || models.length === 0}
              data-testid="model-select"
              aria-label={t("model.saved")}
            >
              <option value="">
                {models.length ? t("model.choose") : t("model.noSaved")}
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
              data-testid="model-load"
              disabled={disabled || !selected}
            >
              {t("action.load")}
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
              placeholder={t("model.name")}
              data-testid="model-name"
              aria-label={t("model.name")}
            />
            <button
              className="apply small"
              onClick={() => onSave(name)}
              data-testid="model-save"
              disabled={disabled || !name.trim()}
            >
              {t("action.save")}
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
                {models.length ? t("model.choose") : t("model.noSaved")}
              </option>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
            <a
              className={`apply small ${bundleTarget ? "" : "disabled"}`}
              href={bundleTarget ? bundleHref(bundleTarget) : undefined}
              aria-disabled={!bundleTarget}
              onClick={(e) => {
                if (!bundleTarget) e.preventDefault();
              }}
              download
            >
              {t("model.download")}
            </a>
          </div>
          <p className="arch-note">
            A portable deployment bundle — everything a runtime needs to load
            and run this model, with no training code attached. Install it as a
            standalone command with{" "}
            <code>spikeforge-serve install &lt;file&gt;.spkf</code>, or serve it
            with <code>spikeforge-serve serve --bundle &lt;file&gt;.spkf</code>.
          </p>
        </div>
      )}

      <ConfirmDialog
        open={confirmUnload}
        title={t("model.unloadTitle")}
        message={t("model.unloadMessage")}
        confirmLabel={t("action.unload")}
        onConfirm={() => {
          onNew();
          setConfirmUnload(false);
        }}
        onCancel={() => setConfirmUnload(false)}
      />
    </section>
  );
}
