import type { HubImportPayload } from "../hubTypes";
import { HubCompatBadge } from "./HubCompatBadge";

interface Props {
  result: HubImportPayload;
}

/** An import verdict: badge, promotion outcome, and named mismatches. */
export function HubVerdictView({ result }: Props) {
  const verdict = result.verdict;
  return (
    <div className="hub-verdict-view">
      <div className="hub-verdict-head">
        <HubCompatBadge verdict={verdict.verdict} />
        <span className="hub-verdict-topo">
          {verdict.topology ?? "no preset match"}
        </span>
      </div>
      <div className="hub-verdict-note">
        {result.promoted
          ? `promoted → ${result.destination ?? "store"}`
          : result.reason ?? "not promoted"}
      </div>

      {verdict.mismatches.length > 0 && (
        <table className="readout-table">
          <thead>
            <tr>
              <th>stage</th>
              <th>reason</th>
              <th>expected</th>
              <th>actual</th>
            </tr>
          </thead>
          <tbody>
            {verdict.mismatches.map((item, index) => (
              <tr key={`${item.stage}-${item.reason}-${index}`}>
                <td>{item.stage}</td>
                <td>{item.reason}</td>
                <td>{item.expected}</td>
                <td>{item.actual}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {result.notes.length > 0 && (
        <ul className="notes">
          {result.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
