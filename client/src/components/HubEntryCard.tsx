import type { HubEntryCard as Entry } from "../hubTypes";

interface Props {
  entry: Entry;
  selected: boolean;
  onSelect: (id: string) => void;
  onDownload: (id: string) => void;
  onInspect: (id: string) => void;
  onImport: (id: string) => void;
}

function sizeLabel(bytes: number | null): string {
  if (bytes === null) return "size unknown";
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

/** One hub catalog entry with its availability and actions. */
export function HubEntryCard({
  entry,
  selected,
  onSelect,
  onDownload,
  onInspect,
  onImport,
}: Props) {
  return (
    <li className={`hub-card${selected ? " selected" : ""}`}>
      <button
        type="button"
        className="hub-card-main"
        onClick={() => onSelect(entry.id)}
        aria-pressed={selected}
      >
        <span className="hub-name">
          {entry.name}
          {entry.cached && <span className="hub-cached">cached</span>}
        </span>
        <span className={`hub-badge ${entry.available ? "ok" : "off"}`}>
          {entry.available ? "available" : entry.reason ?? "unavailable"}
        </span>
        <span className="hub-meta">
          {entry.framework} · {entry.kind} ·{" "}
          {sizeLabel(entry.size_bytes)} · {entry.license}
        </span>
      </button>
      {selected && (
        <div className="hub-actions">
          <button type="button" onClick={() => onDownload(entry.id)}>
            download
          </button>
          <button type="button" onClick={() => onInspect(entry.id)}>
            inspect
          </button>
          <button type="button" onClick={() => onImport(entry.id)}>
            import
          </button>
        </div>
      )}
    </li>
  );
}
