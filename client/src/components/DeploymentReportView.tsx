import type { DeploymentReportPayload } from "../targetTypes";
import { DeploymentBuckets } from "./DeploymentBuckets";
import { NirDriftTable } from "./NirDriftTable";
import { formatMetric } from "./nirFormat";
import { formatConstraint } from "./targetFormat";

interface Props {
  report: DeploymentReportPayload | null;
}

/** Caption summarizing the capability buckets of a report. */
function countLine(report: DeploymentReportPayload): string {
  const c = report.nodes.counts;
  return (
    `${c.supported} supported · ${c.substituted} substituted · ` +
    `${c.unsupported} unsupported · ${c.total} total`
  );
}

/** Verdict, node buckets, constraints, and the optional drift check. */
export function DeploymentReportView({ report }: Props) {
  if (report === null) {
    return (
      <div className="panel-note">
        Select a target to see its deployment report.
      </div>
    );
  }

  const validation = report.validation;
  const verdict = report.deployable ? "ok" : "bad";
  return (
    <>
      <div className={`verdict ${verdict}`}>
        {report.deployable ? "deployable" : "not deployable"}
      </div>
      <div className="panel-caption">
        {report.available ? "SDK available" : "SDK missing"} ·{" "}
        {countLine(report)}
      </div>

      <DeploymentBuckets nodes={report.nodes} />

      <table className="readout-table">
        <thead>
          <tr>
            <th>Constraint</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(report.constraints).map(([key, value]) => (
            <tr key={key}>
              <td>{key}</td>
              <td>{formatConstraint(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {validation === null ? (
        <div className="panel-note">
          No drift validation in this report; load a model and refresh to
          compare the reference interpreter against the exported graph.
        </div>
      ) : (
        <>
          <div className="panel-caption">
            validation ·{" "}
            {validation.within_tolerance
              ? "within tolerance"
              : "drift detected"}
          </div>
          {validation.worst && (
            <div className="worst">
              worst · {validation.worst.layer} ·{" "}
              {validation.worst.quantity} ·{" "}
              {formatMetric(validation.worst.value)}
            </div>
          )}
          <NirDriftTable layers={validation.layers} />
        </>
      )}

      {report.notes.length > 0 && (
        <ul className="notes">
          {report.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </>
  );
}
