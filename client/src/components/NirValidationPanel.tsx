import { HELP } from "../helpText";
import type {
  NirValidationPayload,
  NirValidationWorst,
} from "../nirTypes";
import { HelpTip } from "./HelpTip";
import { NirDriftTable } from "./NirDriftTable";
import { formatMetric } from "./nirFormat";

interface Props {
  validation: NirValidationPayload | null;
  onRefresh: () => void;
}

/** Flatten the readout drift block into one caption line. */
function readoutLine(validation: NirValidationPayload): string {
  const r = validation.readout;
  return (
    `max|Δ| ${formatMetric(r.max_abs)} · mean|Δ| ${formatMetric(r.mean_abs)}` +
    ` · rel ${formatMetric(r.relative)} · agree ${formatMetric(r.agreement)}`
  );
}

/** Format the worst offender as "layer · quantity · metric value". */
function worstLine(worst: NirValidationWorst): string {
  const { layer, quantity, metric, value } = worst;
  return `worst · ${layer} · ${quantity} · ${metric} ${formatMetric(value)}`;
}

/** NIR drift validation: verdict, readout, worst offender, notes, layers. */
export function NirValidationPanel({ validation, onRefresh }: Props) {
  const verdict = validation?.within_tolerance ? "ok" : "bad";
  return (
    <div className="panel nir-validation-panel" data-tour="nir-validation">
      <div className="panel-title row-title">
        <span>
          NIR drift validation
          <HelpTip text={HELP.nir_validation} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={onRefresh}
            title="Validate NIR graph"
            aria-label="Validate NIR graph"
          >
            ↻
          </button>
        </span>
      </div>

      {validation === null ? (
        <div className="panel-note">
          No validation run yet. Press ↻ to compare the exported graph against
          the reference interpreter.
        </div>
      ) : (
        <>
          <div className={`verdict ${verdict}`}>
            {validation.within_tolerance
              ? "within tolerance"
              : "drift detected"}
          </div>
          <div className="panel-caption">
            {validation.steps} steps · readout {readoutLine(validation)}
          </div>
          {validation.worst && (
            <div className="worst">{worstLine(validation.worst)}</div>
          )}
          <NirDriftTable layers={validation.layers} />
          {validation.notes.length > 0 && (
            <ul className="notes">
              {validation.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
