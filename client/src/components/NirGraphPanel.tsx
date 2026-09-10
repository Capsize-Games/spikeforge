import { HELP } from "../helpText";
import type { NirGraphPayload } from "../nirTypes";
import { HelpTip } from "./HelpTip";
import { NirGraphView } from "./NirGraphView";

interface Props {
  graph: NirGraphPayload | null;
  onRefresh: () => void;
}

/** Topology graph panel: refresh action, summary, and the layered SVG. */
export function NirGraphPanel({ graph, onRefresh }: Props) {
  const delayed = graph
    ? graph.edges.filter((edge) => edge.delayed).length
    : 0;

  return (
    <div className="panel nir-graph-panel" data-tour="nir-graph">
      <div className="panel-title row-title">
        <span>
          Topology graph
          <HelpTip text={HELP.nir_graph} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={onRefresh}
            title="Export NIR graph"
            aria-label="Export NIR graph"
          >
            ↻
          </button>
        </span>
      </div>

      {graph === null ? (
        <div className="panel-note">
          No graph exported yet. Press ↻ to summarize the active topology.
        </div>
      ) : (
        <>
          <div className="panel-caption">
            {graph.nodes.length} nodes · {graph.edges.length} edges ·{" "}
            {delayed} delayed
          </div>
          <div className="nir-graph-scroll" data-tour="nir-graph-view">
            <NirGraphView graph={graph} />
          </div>
        </>
      )}
    </div>
  );
}
