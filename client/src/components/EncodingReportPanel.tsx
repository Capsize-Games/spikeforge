import { HELP } from "../helpText";
import type { EncodingReportPayload } from "../introspectionTypes";
import { HeatmapCanvas } from "./HeatmapCanvas";
import { HelpTip } from "./HelpTip";
import { firstPlane, formatShape, statEntries } from "./encodingMath";

interface Props {
  report: EncodingReportPayload | null;
  onRefresh: () => void;
}

/** Percentage with one decimal. */
function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Reconstructed image plus the honest approximation and stats readout. */
export function EncodingReportPanel({ report, onRefresh }: Props) {
  const reconstruction = report ? report.reconstruction : null;
  const plane = firstPlane(reconstruction);
  const entries = report ? statEntries(report.stats) : [];

  return (
    <div className="panel encoding-panel" data-tour="encoding-report">
      <div className="panel-title row-title">
        <span>
          Encoding report
          <HelpTip text={HELP.encoding_report} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={onRefresh}
            title="Encode the displayed sample"
            aria-label="Encode the displayed sample"
          >
            ↻
          </button>
        </span>
      </div>

      <div className="panel-caption">
        Tracks the current LEFT encoding controls.
      </div>

      {report === null ? (
        <div className="panel-note">
          No report yet. Press ↻ to encode the displayed sample and decode it
          back to an image.
        </div>
      ) : (
        <>
          <div className="metrics">
            <div>coding {report.coding}</div>
            <div>steps {report.num_steps}</div>
            <div>shape {formatShape(report.shape)}</div>
            <div>firing rate {report.firing_rate.toFixed(3)}</div>
            <div>sparsity {pct(report.sparsity)}</div>
          </div>

          <div className="panel-title subsection">Reconstruction</div>
          {report.reconstruction_supported && reconstruction && plane ? (
            <>
              <HeatmapCanvas
                data={plane}
                palette="binary"
                width={180}
                height={180}
              />
              {reconstruction.length > 1 && (
                <div className="panel-caption">
                  channel 0 of {reconstruction.length}
                </div>
              )}
            </>
          ) : (
            <div className="panel-note">
              Reconstruction is not available for this coding.
            </div>
          )}

          <div className="panel-title subsection">Approximation</div>
          <div className="panel-note">{report.approximation}</div>

          {entries.length > 0 && (
            <>
              <div className="panel-title subsection">Stats</div>
              <div className="table-scroll">
                <table className="readout-table">
                  <tbody>
                    {entries.map((entry) => (
                      <tr key={entry.key}>
                        <th scope="row">{entry.key}</th>
                        <td>{entry.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
