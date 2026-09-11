import { HELP } from "../helpText";
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
          Deployment targets
          <HelpTip text={HELP.targets} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={() => run(target)}
            title="Compile and run the selected target"
            aria-label="Compile and run the selected target"
          >
            ▶
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={onRefresh}
            title="List deployment targets"
            aria-label="List deployment targets"
          >
            ↻
          </button>
        </span>
      </div>

      {list === null ? (
        <div className="panel-note">
          No target registry loaded yet. Press ↻ to list the deployment
          targets and their availability.
        </div>
      ) : (
        <>
          <TargetList
            targets={list.targets}
            selected={target}
            onSelect={select}
          />
          <div className="panel-caption">Deployment report · {target}</div>
          <DeploymentReportView report={shown} />
          <div className="panel-caption">Backend run · {target}</div>
          <BackendRunPanel run={shownRun} />
        </>
      )}
    </div>
  );
}
