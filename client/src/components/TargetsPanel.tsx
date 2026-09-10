import { HELP } from "../helpText";
import { useTargetSelection } from "../hooks/useTargetSelection";
import type {
  DeploymentReportPayload,
  TargetListPayload,
} from "../targetTypes";
import { DeploymentReportView } from "./DeploymentReportView";
import { HelpTip } from "./HelpTip";
import { TargetList } from "./TargetList";

interface Props {
  list: TargetListPayload | null;
  report: DeploymentReportPayload | null;
  onRefresh: () => void;
  onSelectTarget: (name: string) => void;
}

/** Deployment targets: registry list, availability, and a report. */
export function TargetsPanel({
  list,
  report,
  onRefresh,
  onSelectTarget,
}: Props) {
  const names = list ? list.targets.map((item) => item.name) : [];
  const { target, setTarget } = useTargetSelection(names);

  const select = (name: string) => {
    setTarget(name);
    onSelectTarget(name);
  };

  // Only show a report that matches the selected target, so a stale reply can
  // never imply support for a different target's capabilities.
  const shown = report && report.target.name === target ? report : null;

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
        </>
      )}
    </div>
  );
}
