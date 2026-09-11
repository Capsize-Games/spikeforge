import type { BackendRunPayload, RewriteRecord } from "../targetTypes";
import { formatMetric } from "./nirFormat";
import { describeRewrite } from "./targetFormat";

interface Props {
  run: BackendRunPayload | null;
}

/** Flatten a rewrite report's three buckets into one list. */
function records(run: BackendRunPayload): RewriteRecord[] {
  const rewrite = run.rewritten;
  if (rewrite === null) return [];
  return [
    ...rewrite.applied,
    ...rewrite.skipped,
    ...rewrite.unfixable,
  ];
}

/** The executed backend view: status, rewrite, drift, and comparison. */
export function BackendRunPanel({ run }: Props) {
  if (run === null) {
    return (
      <div className="panel-note">
        Run the selected target to compile the graph and see an executed
        result. An absent SDK is reported honestly instead of hidden.
      </div>
    );
  }

  const verdict = run.status === "ok" ? "ok" : "bad";
  const rewrite = run.rewritten;
  const items = records(run);
  return (
    <>
      <div className={`verdict ${verdict}`}>{run.status}</div>
      <div className="panel-caption">
        {run.target}
        {run.path ? ` · ${run.path}` : ""} · {run.steps} steps
      </div>

      {run.compare && (
        <div className="panel-caption">
          vs reference · readout{" "}
          {formatMetric(run.compare.readout.max_abs)} · spikes{" "}
          {formatMetric(run.compare.spikes.max_abs)}
        </div>
      )}

      {rewrite && (
        <div className="backend-rewrite">
          <div className="panel-caption">
            rewrite · {rewrite.counts.applied} applied ·{" "}
            {rewrite.counts.skipped} skipped ·{" "}
            {rewrite.counts.unfixable} unfixable
          </div>
          {items.length === 0 ? (
            <div className="bucket-empty">no substitutions needed</div>
          ) : (
            <ul className="notes">
              {items.map((item, index) => (
                <li key={`${item.node}-${index}`}>
                  {describeRewrite(item)}
                </li>
              ))}
            </ul>
          )}
          {rewrite.drift && (
            <div className="panel-caption">
              drift ·{" "}
              {rewrite.drift.within_tolerance
                ? "within tolerance"
                : "drift detected"}
            </div>
          )}
        </div>
      )}

      {run.notes.length > 0 && (
        <ul className="notes">
          {run.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </>
  );
}
