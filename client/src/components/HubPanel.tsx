import { useState } from "react";

import { HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import { HUB_FRAMEWORKS, HUB_KINDS } from "../hubTypes";
import { useHub } from "../hooks/useHub";
import { HelpTip } from "./HelpTip";
import { HubDownloadProgress } from "./HubDownloadProgress";
import { HubEntryCard } from "./HubEntryCard";
import { HubVerdictView } from "./HubVerdictView";

type Hub = ReturnType<typeof useHub>;

interface Props {
  hub: Hub;
}

/** The model-hub browser: filters, entry cards, downloads, and imports. */
export function HubPanel({ hub }: Props) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [framework, setFramework] = useState("");
  const [kind, setKind] = useState("");

  const entries = hub.search ? hub.search.results : (hub.list?.entries ?? []);
  const issues = hub.list?.issues ?? [];
  const active = hub.download?.status === "downloading" ? hub.download : null;

  const changeFramework = (value: string) => {
    setFramework(value);
    hub.refresh({ framework: value || null, kind: kind || null });
  };

  const changeKind = (value: string) => {
    setKind(value);
    hub.refresh({ framework: framework || null, kind: value || null });
  };

  return (
    <div className="panel hub-panel" data-tour="hub">
      <div className="panel-title row-title">
        <span>
          {t("hub.title")}
          <HelpTip text={HELP.hub} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={() => hub.refresh()}
            title={t("hub.refresh")}
            aria-label={t("hub.refresh")}
          >
            ↻
          </button>
        </span>
      </div>

      <div className="hub-filters">
        <input
          className="hub-search-input"
          value={query}
          placeholder={t("hub.searchPlaceholder")}
          aria-label={t("hub.search")}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button type="button" onClick={() => hub.runSearch(query)}>
          {t("hub.search")}
        </button>
        <select
          value={framework}
          aria-label={t("hub.framework")}
          onChange={(event) => changeFramework(event.target.value)}
        >
          <option value="">{t("hub.allFrameworks")}</option>
          {HUB_FRAMEWORKS.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <select
          value={kind}
          aria-label={t("hub.kind")}
          onChange={(event) => changeKind(event.target.value)}
        >
          <option value="">{t("hub.allKinds")}</option>
          {HUB_KINDS.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      {hub.search && (
        <div className="hub-search-caption">
          search “{hub.search.query}” · live search{" "}
          {hub.search.available ? "available" : "unavailable"}
          {hub.search.reason ? ` (${hub.search.reason})` : ""}
        </div>
      )}

      {issues.length > 0 && (
        <ul className="hub-issues">
          {issues.map((issue) => (
            <li key={issue}>{issue}</li>
          ))}
        </ul>
      )}

      {entries.length === 0 ? (
        <div className="panel-note">{t("hub.empty")}</div>
      ) : (
        <ul className="hub-list">
          {entries.map((entry) => (
            <HubEntryCard
              key={entry.id}
              entry={entry}
              selected={hub.selected === entry.id}
              onSelect={hub.select}
              onDownload={hub.downloadEntry}
              onInspect={hub.inspectEntry}
              onImport={hub.importEntry}
            />
          ))}
        </ul>
      )}

      {active && (
        <HubDownloadProgress download={active} onCancel={hub.cancelDownload} />
      )}

      {hub.inspect && (
        <div className="hub-inspect">
          <div className="panel-caption">
            inspect · {hub.inspect.kind} · {hub.inspect.nodes.length} nodes
          </div>
          {hub.inspect.notes.map((note) => (
            <div key={note} className="hub-note">
              {note}
            </div>
          ))}
        </div>
      )}

      {hub.imported && <HubVerdictView result={hub.imported} />}
    </div>
  );
}
