import type { TargetSummary } from "../targetTypes";

interface Props {
  targets: TargetSummary[];
  selected: string;
  onSelect: (name: string) => void;
}

/** Selectable target rows with an honest availability badge. */
export function TargetList({ targets, selected, onSelect }: Props) {
  return (
    <ul className="target-list">
      {targets.map((item) => {
        const active = item.name === selected;
        const extra = item.extra ? `extra: ${item.extra}` : "no extra";
        return (
          <li key={item.name}>
            <button
              type="button"
              className={`target-item${active ? " selected" : ""}`}
              onClick={() => onSelect(item.name)}
              aria-pressed={active}
            >
              <span className="target-name">
                {item.name}
                <span className="target-kind">{item.kind}</span>
              </span>
              <span
                className={`target-badge ${item.available ? "ok" : "off"}`}
              >
                {item.available ? "available" : "SDK missing"}
              </span>
              <span className="target-meta">
                {item.supported_count} primitives · {extra}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
