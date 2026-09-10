import type { DeviceChoice, MemoryBlock, SystemStatsPayload } from "../types";

interface Props {
  stats: SystemStatsPayload | null;
  requested: DeviceChoice;
}

function gb(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(1);
}

function ResourceItem({
  label,
  block,
}: {
  label: string;
  block: MemoryBlock | null;
}) {
  if (!block) {
    return (
      <div className="res-item">
        <span className="res-label">{label}</span>
        <span className="res-muted">n/a</span>
      </div>
    );
  }
  return (
    <div className="res-item" title={block.name ?? label}>
      <span className="res-label">{label}</span>
      <span className="res-value">
        {gb(block.used)}/{gb(block.total)} GB
      </span>
      <span className="res-track">
        <span
          className="res-fill"
          style={{ width: `${Math.min(block.percent, 100)}%` }}
        />
      </span>
      <span className="res-muted">{block.percent.toFixed(0)}%</span>
    </div>
  );
}

export function ResourceMonitor({ stats, requested }: Props) {
  const device = stats?.device;
  const gpuAvailable = device?.available.includes("gpu") ?? false;
  const active = device?.active;
  const gpuWarning = requested === "gpu" && !gpuAvailable;

  return (
    <div className="res-bar">
      <ResourceItem label="CPU RAM" block={stats?.cpu ?? null} />
      <ResourceItem label="GPU VRAM" block={stats?.gpu ?? null} />
      <div className="res-item">
        <span className="res-label">Device</span>
        <span className="res-value">
          {requested === "auto" ? "AUTO" : requested.toUpperCase()}
          {active ? ` → ${active === "cuda" ? "GPU" : "CPU"}` : ""}
        </span>
      </div>
      {gpuWarning && (
        <span className="res-warn">GPU unavailable — using CPU</span>
      )}
      {!stats && <span className="res-muted">waiting for stats…</span>}
    </div>
  );
}
