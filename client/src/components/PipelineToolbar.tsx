import { useState } from "react";

import { ConfirmDialog } from "./ConfirmDialog";
import type { PipelineGraph } from "../pipelineTypes";
import type { SavedModel } from "../types";

interface Props {
  models: SavedModel[];
  pipelines: { name: string; node_count: number; edge_count: number }[];
  graphName: string | undefined;
  nodeCount: number;
  edgeCount: number;
  initialName: string;
  onListPipelines: () => void;
  onSavePipeline: (name: string) => void;
  onLoadPipeline: (name: string) => void;
  onDeletePipeline: (name: string) => void;
  onSetGraph: (graph: PipelineGraph) => void;
  onAddCheckpoint: (checkpoint: string) => void;
}

/** Name/save, saved-pipeline load/delete, and add-node controls. */
export function PipelineToolbar({
  models,
  pipelines,
  graphName,
  nodeCount,
  edgeCount,
  initialName,
  onListPipelines,
  onSavePipeline,
  onLoadPipeline,
  onDeletePipeline,
  onSetGraph,
  onAddCheckpoint,
}: Props) {
  const [name, setName] = useState(initialName);
  const [selected, setSelected] = useState("");
  const [confirmAction, setConfirmAction] = useState<
    "new" | "delete" | null
  >(null);

  return (
    <div className="pipeline-toolbar panel">
      <div className="control-row">
        <input
          className="text-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="pipeline name…"
          data-testid="pipeline-name"
          aria-label="Pipeline name"
        />
        <button
          className="apply small"
          onClick={() => name.trim() && onSavePipeline(name.trim())}
          data-testid="pipeline-save"
          disabled={!name.trim()}
        >
          Save
        </button>
        <button
          className="apply small ghost"
          onClick={() => setConfirmAction("new")}
          data-testid="pipeline-new"
          disabled={nodeCount === 0 && edgeCount === 0}
        >
          New
        </button>
      </div>

      <div className="control-row">
        <select
          className="text-input"
          data-testid="pipeline-saved"
          aria-label="Saved pipelines"
          onFocus={onListPipelines}
          onChange={(e) => e.target.value && onLoadPipeline(e.target.value)}
          value=""
        >
          <option value="">
            {pipelines.length ? "load a saved pipeline…" : "no saved pipelines"}
          </option>
          {pipelines.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} ({p.node_count} nodes)
            </option>
          ))}
        </select>
        <button
          className="apply small ghost"
          onClick={() => graphName && setConfirmAction("delete")}
          data-testid="pipeline-delete"
          disabled={!graphName}
        >
          Delete
        </button>
      </div>

      <div className="control-row">
        <select
          className="text-input"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={models.length === 0}
          data-testid="pipeline-checkpoint"
          aria-label="Checkpoint to add"
        >
          <option value="">
            {models.length ? "choose a saved model…" : "no saved models"}
          </option>
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
        <button
          className="apply small"
          onClick={() => selected && onAddCheckpoint(selected)}
          data-testid="pipeline-add-node"
          disabled={!selected}
        >
          Add node
        </button>
      </div>

      <ConfirmDialog
        open={confirmAction === "new"}
        title="Discard this pipeline?"
        message={
          "This clears the current graph. Save it first if you want to keep it."
        }
        confirmLabel="Discard"
        onConfirm={() => {
          onSetGraph({ version: 1, name: "", nodes: [], edges: [] });
          setConfirmAction(null);
        }}
        onCancel={() => setConfirmAction(null)}
      />
      <ConfirmDialog
        open={confirmAction === "delete"}
        title={`Delete "${graphName}"?`}
        message={
          "This permanently removes the saved pipeline. This can't be undone."
        }
        confirmLabel="Delete"
        onConfirm={() => {
          if (graphName) onDeletePipeline(graphName);
          setConfirmAction(null);
        }}
        onCancel={() => setConfirmAction(null)}
      />
    </div>
  );
}
