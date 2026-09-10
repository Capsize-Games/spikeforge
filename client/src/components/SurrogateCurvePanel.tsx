import { useSurrogateCurve } from "../hooks/useSurrogateCurve";
import { HELP } from "../helpText";
import type { SurrogateCurvePayload } from "../introspectionTypes";
import { useTheme } from "../theme";
import { HelpTip } from "./HelpTip";
import { LineChart } from "./LineChart";

interface Props {
  /** Selectable surrogate registry names. */
  names: string[];
  /** The currently configured surrogate, "" for the built-in default. */
  current: string;
  curve: SurrogateCurvePayload | null;
  onRequest: (name: string) => void;
}

/** Fixed-precision axis bound. */
function edge(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(1);
}

/** Derivative curve of the selected surrogate gradient. */
export function SurrogateCurvePanel({
  names,
  current,
  curve,
  onRequest,
}: Props) {
  const { colors } = useTheme();
  const { value, setValue, reload } = useSurrogateCurve(current, onRequest);
  // Ignore a stale curve so the plot never mislabels a different surrogate.
  const shown = curve && curve.name === value ? curve : null;

  return (
    <div className="panel surrogate-panel" data-tour="surrogate-curve">
      <div className="panel-title row-title">
        <span>
          Surrogate derivative
          <HelpTip text={HELP.surrogate_curve} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={reload}
            disabled={value === ""}
            title="Sample the derivative"
            aria-label="Sample the derivative"
          >
            ↻
          </button>
        </span>
      </div>

      <label className="trajectory-stage">
        <span className="trajectory-stage-label">Surrogate</span>
        <select
          value={value}
          onChange={(event) => setValue(event.target.value)}
          aria-label="Surrogate gradient"
        >
          <option value="">Default (snnTorch)</option>
          {names.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      {value === "" ? (
        <div className="panel-note">
          The default uses snnTorch's built-in Fast Sigmoid; pick a named
          surrogate to visualise its derivative.
        </div>
      ) : shown === null ? (
        <div className="panel-note">
          No curve loaded yet. Press ↻ to sample {value}.
        </div>
      ) : (
        <>
          <div className="surrogate-chart">
            <LineChart
              series={[
                {
                  label: `dS/dU · ${shown.name}`,
                  color: colors.accent,
                  values: shown.y,
                },
              ]}
              width={340}
              height={150}
              bare
            />
          </div>
          <div className="panel-caption">
            {shown.y.length} samples · x ∈ [{edge(shown.x[0])},{" "}
            {edge(shown.x[shown.x.length - 1])}]
          </div>
        </>
      )}
    </div>
  );
}
