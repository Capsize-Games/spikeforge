import { HelpTip } from "./HelpTip";
import type { PipelineEdge } from "../usePipeline";

interface Props {
  selectedEdge: string;
  extract: NonNullable<PipelineEdge["extract"]>;
  onSetEdgeExtract: (
    id: string,
    extract: NonNullable<PipelineEdge["extract"]>,
  ) => void;
}

const EXTRACT_HELP =
  "How the source node's output is shaped into the target node's next input.";

/** Inspector for the selected edge's output-shaping mode. */
export function PipelineEdgeInspector({
  selectedEdge,
  extract,
  onSetEdgeExtract,
}: Props) {
  return (
    <div className="pipeline-edge-inspector">
      <div className="panel-title row-title">
        <span>Edge</span>
        <HelpTip text={EXTRACT_HELP} />

      </div>
      <select
        className="text-input"
        value={extract}
        onChange={(e) =>
          onSetEdgeExtract(
            selectedEdge,
            e.target.value as NonNullable<PipelineEdge["extract"]>,
          )
        }
      >
        <option value="mean_logits">mean logits vector</option>
        <option value="predicted_class">predicted class (scalar)</option>
        <option value="one_hot">one-hot of predicted class</option>
      </select>
    </div>
  );
}
