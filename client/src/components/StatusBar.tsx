import type { DeviceChoice, SystemStatsPayload } from "../types";
import { ResourceMonitor } from "./ResourceMonitor";

interface Props {
  stats: SystemStatsPayload | null;
  requested: DeviceChoice;
  connected: boolean;
}

/** Footer: the resource monitor plus the live connection indicator. */
export function StatusBar({ stats, requested, connected }: Props) {
  return (
    <footer className="app-footer">
      <ResourceMonitor stats={stats} requested={requested} />
      <span className={`conn ${connected ? "ok" : "bad"}`}>
        <span className={`dot ${connected ? "ok" : "bad"}`} />
        {connected ? "connected" : "disconnected"}
      </span>
    </footer>
  );
}
