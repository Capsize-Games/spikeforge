import type { DeviceChoice, SystemStatsPayload } from "../types";
import { useI18n } from "../i18n/I18nProvider";
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
  const { t } = useI18n();
  const label = unauthorized
    ? t("status.unauthorized")
    : connected
      ? t("status.connected")
      : t("status.disconnected");
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
