import { HELP } from "../helpText";
import { useSelectedStage } from "../hooks/useSelectedStage";
import type {
  StageMetrics,
  TrajectoryMetricsPayload,
} from "../introspectionTypes";
import type { ExecutionMode } from "../types";
import { HelpTip } from "./HelpTip";
import { HistogramChart } from "./HistogramChart";
import { TrajectoryStageSelect } from "./TrajectoryStageSelect";

interface Props {
  mode: ExecutionMode;
  metrics: TrajectoryMetricsPayload | null;
  onRefresh: () => void;
  onSwitchMode: () => void;
}

/** Percentage with one decimal. */
function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** A metric that may be null when it is undefined for the trace. */
function num(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(3);
}

/** The per-stage readout grid: rate, sparsity, and ISI statistics. */
function MetricGrid({ metrics }: { metrics: StageMetrics }) {
  const isi = metrics.isi;
  return (
    <div className="metrics">
      <div>firing rate {num(metrics.firing_rate)}</div>
      <div>sparsity {pct(metrics.sparsity)}</div>
      <div>ISI count {isi.count}</div>
      <div>ISI mean {num(isi.mean)}</div>
      <div>ISI median {num(isi.median)}</div>
      <div>ISI std {num(isi.std)}</div>
      <div>ISI cv {num(isi.cv)}</div>
    </div>
  );
}

/** Per-stage firing-rate/sparsity/ISI metrics with a mode gate. */
export function MetricsPanel({
  mode,
  metrics,
  onRefresh,
  onSwitchMode,
}: Props) {
  const educational = mode === "educational";
  const stages = metrics ? Object.keys(metrics.stages) : [];
  const { stage, setStage } = useSelectedStage(stages);
  const selected = metrics ? metrics.stages[stage] ?? null : null;

  return (
    <div className="panel metrics-panel">
      <div className="panel-title row-title">
        <span>
          Trajectory metrics
          <HelpTip text={HELP.metrics} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={onRefresh}
            disabled={!educational}
            title="Capture metrics"
            aria-label="Capture metrics"
          >
            ↻
          </button>
        </span>
      </div>

      {!educational && (
        <div className="panel-note">
          Aggregate metrics need educational mode, which records the
          per-step spikes they are derived from.{" "}
          <button type="button" className="link" onClick={onSwitchMode}>
            Switch to educational
          </button>
        </div>
      )}

      {educational && metrics === null && (
        <div className="panel-note">
          No metrics captured yet. Press ↻ to aggregate the active model's
          firing rate, sparsity, and inter-spike intervals.
        </div>
      )}

      {educational && metrics !== null && (
        <>
          <TrajectoryStageSelect
            stages={stages}
            value={stage}
            onChange={setStage}
          />
          {selected === null ? (
            <div className="panel-note">This stage has no metrics.</div>
          ) : (
            <>
              <MetricGrid metrics={selected} />
              <div className="panel-title subsection">
                Firing-rate histogram
              </div>
              <HistogramChart
                edges={selected.histogram.edges}
                counts={selected.histogram.counts}
              />
            </>
          )}
          <div className="panel-caption">
            {metrics.steps} steps · {stages.length} stages
          </div>
        </>
      )}
    </div>
  );
}
