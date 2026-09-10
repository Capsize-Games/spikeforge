import { HELP } from "../helpText";
import { useSelectedStage } from "../hooks/useSelectedStage";
import type { TrajectoryPayload } from "../introspectionTypes";
import type { ExecutionMode } from "../types";
import { HelpTip } from "./HelpTip";
import { TrajectoryChart } from "./TrajectoryChart";
import { TrajectoryStageSelect } from "./TrajectoryStageSelect";

interface Props {
  mode: ExecutionMode;
  trajectory: TrajectoryPayload | null;
  cursorIndex: number | null;
  onRefresh: () => void;
  onSwitchMode: () => void;
}

/** Honest caption: how much was captured and which caps truncated it. */
function caption(payload: TrajectoryPayload): string {
  const { steps, caps, stages } = payload;
  return (
    `${steps} steps captured · ${stages.length}/${caps.max_stages} stages ` +
    `· cap ${caps.max_neurons} neurons/stage`
  );
}

/** Per-stage U[t] membrane and I[t] current viewer with a mode gate. */
export function TrajectoryPanel({
  mode,
  trajectory,
  cursorIndex,
  onRefresh,
  onSwitchMode,
}: Props) {
  const educational = mode === "educational";
  const stages = trajectory ? trajectory.stages : [];
  const { stage, setStage } = useSelectedStage(stages);
  const trace = trajectory ? trajectory.traces[stage] ?? null : null;

  return (
    <div className="panel trajectory-panel" data-tour="trajectory">
      <div className="panel-title row-title">
        <span>
          Neuron state
          <HelpTip text={HELP.trajectory} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={onRefresh}
            disabled={!educational}
            title="Capture trajectory"
            aria-label="Capture trajectory"
          >
            ↻
          </button>
        </span>
      </div>

      {!educational && (
        <div className="panel-note">
          Per-step U[t]/I[t] capture is disabled in production mode.{" "}
          <button type="button" className="link" onClick={onSwitchMode}>
            Switch to educational
          </button>
        </div>
      )}

      {educational && trajectory === null && (
        <div className="panel-note">
          No trajectory captured yet. Press ↻ to record the active model's
          membrane and input current.
        </div>
      )}

      {educational && trajectory !== null && (
        <>
          <TrajectoryStageSelect
            stages={trajectory.stages}
            value={stage}
            onChange={setStage}
          />
          <div data-tour="trajectory-chart">
            <TrajectoryChart
              trace={trace}
              cursorIndex={cursorIndex}
              emptyNote="This stage has no captured trace."
            />
          </div>
          <div className="panel-caption">{caption(trajectory)}</div>
        </>
      )}
    </div>
  );
}
