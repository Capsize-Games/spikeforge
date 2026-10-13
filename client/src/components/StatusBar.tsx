import type { DeviceChoice, SystemStatsPayload } from "../types";
import { useI18n } from "../i18n/I18nProvider";
import { ResourceMonitor } from "./ResourceMonitor";
import { StatusIndicator } from "./StatusIndicator";

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
      <a
        className="privacy-link"
        href="https://spikeforge.net/privacy/"
        target="_blank"
        rel="noopener noreferrer"
      >
        {t("footer.privacy")}
      </a>
      <StatusIndicator
        ok={connected}
        label={label}
        className="conn"
        testId="connection"
      />
    </footer>
  );
}
