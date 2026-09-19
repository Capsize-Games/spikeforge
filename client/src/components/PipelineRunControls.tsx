import { useState } from "react";

import { HelpTip } from "./HelpTip";
import { inputContract, inputTemplate } from "./pipelineInput";
import type { PipelineRunInput } from "../pipelineTypes";
import type { SavedModel } from "../types";

const INPUT_HELP =
  "Feeds every node with no incoming edge. The default uses encoded: false " +
  "with one raw [channels, height, width] sample, which the server encodes " +
  "using the checkpoint's metadata. With encoded: true, each frame must " +
  "already match the topology's input shape.";

interface Props {
  /** Checkpoints a run feeds: the nodes with no incoming edge. */
  sources: SavedModel[];
  connected: boolean;
  running: boolean;
  /** True once the graph has a node, so a run has something to feed. */
  runnable: boolean;
  status: string | null;
  onRun: (input: PipelineRunInput) => void;
  onStop: () => void;
}

/**
 * The run body and the start/stop actions.
 *
 * The body is seeded from `source`'s own metadata (see `pipelineInput`) so the
 * placeholder is always a request that runs, and re-seeded whenever the source
 * changes -- but only while it is still a template. Once the user has typed,
 * the body is theirs and is never overwritten.
 */
export function PipelineRunControls({
  sources,
  connected,
  running,
  runnable,
  status,
  onRun,
  onStop,
}: Props) {
  const source = sources[0];
  const [text, setText] = useState(() => inputTemplate(source));
  const [error, setError] = useState<string | null>(null);
  const [edited, setEdited] = useState(false);
  const [seededFor, setSeededFor] = useState(source?.name ?? null);

  // Re-seed during render rather than in an effect: the new template is
  // derived state, and an effect would render one frame of a stale body.
  if (!edited && (source?.name ?? null) !== seededFor) {
    setSeededFor(source?.name ?? null);
    setText(inputTemplate(source));
  }

  // One body feeds every source node, so sources that disagree about their
  // input cannot all be satisfied. Say so rather than seeding from whichever
  // one happens to come first and letting the rest fail inside Torch.
  const contracts = [...new Set(sources.map(inputContract))];
  const mismatched = contracts.length > 1;

  const run = () => {
    let parsed: PipelineRunInput;
    try {
      parsed = JSON.parse(text) as PipelineRunInput;
    } catch {
      setError("input is not valid JSON");
      return;
    }
    if (parsed === null || !Array.isArray(parsed.frames)) {
      setError("input must have a 'frames' array");
      return;
    }
    setError(null);
    onRun(parsed);
  };

  return (
    <>
      <div className="panel-title row-title">
        <span>Run</span>
        <HelpTip text={INPUT_HELP} />
      </div>
      <textarea
        className="text-input pipeline-input"
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          setEdited(true);
        }}
        rows={5}
        data-testid="pipeline-input"
        aria-label="Pipeline run input (JSON)"
      />
      {mismatched && (
        <div className="mismatch" data-testid="pipeline-source-mismatch">
          This graph has source nodes with different input shapes (
          {contracts.join(", ")}). One run body cannot satisfy all of them;
          give them separate runs, or chain them so only one takes the input.
        </div>
      )}
      {error && <div className="mismatch">{error}</div>}
      <div className="control-row">
        <button
          className="apply"
          onClick={run}
          data-testid="pipeline-run"
          disabled={!connected || running || !runnable}
        >
          {running ? "Running…" : "Run pipeline"}
        </button>
        <button
          className="apply stop small"
          onClick={onStop}
          data-testid="pipeline-stop"
          disabled={!running}
        >
          Stop
        </button>
      </div>
      {status && (
        <div className="muted" data-testid="pipeline-status">
          {status}
        </div>
      )}
    </>
  );
}
