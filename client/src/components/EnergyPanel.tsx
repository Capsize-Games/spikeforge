import type { EnergyPayload } from "../energyTypes";
import { useI18n } from "../i18n/I18nProvider";
import { HELP } from "../helpText";
import { useTargetSelection } from "../hooks/useTargetSelection";
import type { TargetSummary } from "../targetTypes";
import { Button } from "./Button";
import { PanelHeader } from "./PanelHeader";

interface Props {
  payload: EnergyPayload | null;
  targets: TargetSummary[];
  loading: boolean;
  onRun: (target: string) => void;
}

/** Thousands-separated integer count. */
function count(value: number): string {
  return value.toLocaleString("en-US");
}

/** A picojoule total rendered with an adaptive unit. */
function energy(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(3)} µJ`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(3)} nJ`;
  return `${value.toFixed(2)} pJ`;
}

/** A nanosecond total rendered with an adaptive unit. */
function latency(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(3)} ms`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(3)} µs`;
  return `${value.toFixed(2)} ns`;
}

/** A ratio with four decimals, or an em dash when undefined. */
function ratio(value: number | null): string {
  return value === null ? "—" : value.toFixed(4);
}

/** The report body: ops, energy, latency, parity, and source notes. */
function Report({ payload }: { payload: EnergyPayload }) {
  const { report, comparison } = payload;
  const badge = report.estimate ? "estimate" : "measured";
  return (
    <>
      <div className="panel-caption">
        {report.target} ·{" "}
        <span className={`energy-badge ${badge}`}>{badge}</span> ·{" "}
        {report.basis}
      </div>

      <div className="energy-grid" data-testid="energy-report">
        <span className="energy-key">SOP</span>
        <span>{count(report.ops.sop)}</span>
        <span className="energy-key">MAC</span>
        <span>{count(report.ops.mac)}</span>
        <span className="energy-key">AC</span>
        <span>{count(report.ops.ac)}</span>
        <span className="energy-key">SOP/MAC</span>
        <span>{ratio(report.efficiency.sop_over_mac)}</span>
        <span className="energy-key">timesteps</span>
        <span>{report.timesteps ?? "—"}</span>
      </div>

      {report.energy === null ? (
        <div className="panel-note">
          energy unavailable for this target. No cost table is declared, so
          nothing is estimated or fabricated.
        </div>
      ) : (
        <div className="panel-caption">
          energy {energy(report.energy.total_pj)} · dense{" "}
          {energy(report.energy.dense_pj)}
        </div>
      )}

      {report.latency !== null && (
        <div className="panel-caption">
          latency {latency(report.latency.total_ns)}
        </div>
      )}

      <div className="panel-caption">
        sparse vs dense · max |Δ| {comparison.max_abs.toExponential(2)} ·{" "}
        {comparison.within_tolerance ? "within tolerance" : "drift"}
      </div>

      <ul className="notes">
        {report.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </>
  );
}

/** The event-driven energy/latency estimate with an explicit run button. */
export function EnergyPanel({ payload, targets, loading, onRun }: Props) {
  const { t } = useI18n();
  const names = targets.map((item) => item.name);
  const { target, setTarget } = useTargetSelection(names);

  return (
    <div className="panel energy-panel" data-tour="energy">
      <PanelHeader
        title={t("energy.title")}
        hint={HELP.energy}
        actions={
          <Button
            variant="primary"
            small
            testId="energy-run"
            onClick={() => onRun(target)}
            disabled={loading || names.length === 0}
          >
            {loading ? t("energy.estimating") : t("energy.estimate")}
          </Button>
        }
      />

      <select
        className="energy-target"
        data-testid="energy-target"
        value={target}
        aria-label={t("energy.target")}
        onChange={(event) => setTarget(event.target.value)}
      >
        {names.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>

      {payload === null ? (
        <div className="panel-note">{t("energy.empty")}</div>
      ) : (
        <Report payload={payload} />
      )}
    </div>
  );
}
