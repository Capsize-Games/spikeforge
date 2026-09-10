import { TRAIN_HELP } from "../helpText";
import type { DeviceChoice, MemoryBlock, SystemStatsPayload } from "../types";
import { HelpTip } from "./HelpTip";

interface Props {
  stats: SystemStatsPayload | null;
  requested: DeviceChoice;
}

function gb(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(1);
}

function MemoryBar({
  label,
  block,
}: {
  label: string;
  block: MemoryBlock | null;
}) {
  if (!block) {
    return (
      <div className="resource-row">
        <div className="resource-head">
          <span>{label}</span>
          <span className="muted">unavailable</span>
        </div>
      </div>
    );
  }
  return (
    <div className="resource-row">
      <div className="resource-head">
        <span>
          {label}
          {block.name ? ` · ${block.name}` : ""}
        </span>
        <span>
          {gb(block.used)} / {gb(block.total)} GB in use · {gb(block.available)}{" "}
          GB free
        </span>
      </div>
      <div className="resource-bar">
        <div
          className="resource-fill"
          style={{ width: `${Math.min(block.percent, 100)}%` }}
        />
      </div>
      <div className="resource-pct">{block.percent.toFixed(1)}% used</div>
    </div>
  );
}

export function ResourceMonitor({ stats, requested }: Props) {
  const device = stats?.device;
  const gpuAvailable = device?.available.includes("gpu") ?? false;
  const active = device?.active;
  return (
    <div className="panel resource-monitor">
      <div className="panel-title">
        <span>System resources</span>
        <HelpTip text={TRAIN_HELP.resources} />
      </div>
      <MemoryBar label="CPU RAM" block={stats?.cpu ?? null} />
      <MemoryBar label="GPU VRAM" block={stats?.gpu ?? null} />
      <div className="resource-device">
        selected <b>{requested.toUpperCase()}</b>
        {active ? (
          <>
            {" "}
            · active <b>{active === "cuda" ? "GPU" : "CPU"}</b>
          </>
        ) : null}
      </div>
      {requested === "gpu" && !gpuAvailable && (
        <div className="mismatch">
          GPU unavailable in this runtime — training falls back to CPU.
        </div>
      )}
      {!stats && <div className="muted">waiting for stats…</div>}
    </div>
  );
}
