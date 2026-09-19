import { HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import { useTargetSelection } from "../hooks/useTargetSelection";
import type {
  BackendRunPayload,
  DeploymentReportPayload,
  TargetListPayload,
} from "../targetTypes";
import { BackendRunPanel } from "./BackendRunPanel";
import { DeploymentReportView } from "./DeploymentReportView";
import { HelpTip } from "./HelpTip";
import { TargetList } from "./TargetList";

interface Props {
  list: TargetListPayload | null;
  report: DeploymentReportPayload | null;
  backendRun: BackendRunPayload | null;
  onRefresh: () => void;
  onSelectTarget: (name: string) => void;
  onRunBackend: (name: string) => void;
}

/** Deployment targets: registry, capability report, and an executed run. */
export function TargetsPanel({
  list,
  report,
  backendRun,
  onRefresh,
  onSelectTarget,
  onRunBackend,
}: Props) {
  const { t } = useI18n();
  const names = list ? list.targets.map((item) => item.name) : [];
  const { target, setTarget } = useTargetSelection(names);

  const select = (name: string) => {
    setTarget(name);
    onSelectTarget(name);
  };

  const run = (name: string) => {
    setTarget(name);
    onSelectTarget(name);
    onRunBackend(name);
  };

  // Only show a report that matches the selected target, so a stale reply can
  // never imply support for a different target's capabilities.
  const shown = report && report.target.name === target ? report : null;
  const shownRun =
    backendRun && backendRun.target === target ? backendRun : null;

  return (
    <div className="panel targets-panel" data-tour="targets">
      <div className="panel-title row-title">
        <span>
          {t("targets.title")}
          <HelpTip text={HELP.targets} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={() => run(target)}
            data-testid="targets-run"
            title={t("targets.run")}
            aria-label={t("targets.run")}
          >
            ▶
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={onRefresh}
            data-testid="targets-refresh"
            title={t("targets.refresh")}
            aria-label={t("targets.refresh")}
          >
            ↻
          </button>
        </span>
      </div>

      {list === null ? (
        <div className="panel-note">
          No target registry loaded yet. Press ↻ to list the deployment targets
          and their availability.
        </div>
      ) : (
        <>
          <TargetList
            targets={list.targets}
            selected={target}
            onSelect={select}
          />
          <div className="panel-caption">
            {t("targets.report")} · {target}
          </div>
          <DeploymentReportView report={shown} />
          <div className="panel-caption">
            {t("targets.backend")} · {target}
          </div>
          <BackendRunPanel run={shownRun} />
        </>
      )}
    </div>
  );
}
