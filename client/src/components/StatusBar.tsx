import type { DeviceChoice, SystemStatsPayload } from "../types";
import { ResourceMonitor } from "./ResourceMonitor";

interface Props {
  stats: SystemStatsPayload | null;
  requested: DeviceChoice;
  connected: boolean;
  unauthorized?: boolean;
}

/** Footer: the resource monitor plus the live connection indicator. */
export function StatusBar({
  stats,
  requested,
  connected,
  unauthorized,
}: Props) {
  const label = unauthorized
    ? "unauthorized — missing or invalid access token"
    : connected
      ? "connected"
      : "disconnected";
  return (
    <footer className="app-footer">
      <ResourceMonitor stats={stats} requested={requested} />
      <span className={`conn ${connected ? "ok" : "bad"}`}>
        <span className={`dot ${connected ? "ok" : "bad"}`} />
        {label}
      </span>
    </footer>
  );
}
