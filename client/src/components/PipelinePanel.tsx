import { useCallback, useMemo, useState } from "react";
import {
  Background,
  Controls as FlowControls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
} from "@xyflow/react";
import type {
  Connection,
  Edge,
  Node,
  NodeChange,
  EdgeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { HelpTip } from "./HelpTip";
import {
  PipelineFlowNode,
  type PipelineFlowNodeData,
} from "./PipelineFlowNode";
import type { PipelineEdge, PipelineNode } from "../usePipeline";
import type {
  PipelineGraph,
  PipelineNodeStatus,
  PipelineRunInput,
} from "../pipelineTypes";
import type { SavedModel } from "../types";

const EXTRACT_LABEL: Record<string, string> = {
  mean_logits: "mean logits vector",
  predicted_class: "predicted class (scalar)",
  one_hot: "one-hot of predicted class",
};

const NODE_TYPES = { checkpoint: PipelineFlowNode };

interface Props {
  models: SavedModel[];
  connected: boolean;
  graph: PipelineGraph;
  pipelines: { name: string; node_count: number; edge_count: number }[];
  running: boolean;
  nodeStatus: Record<string, PipelineNodeStatus>;
  nodeResult: Record<string, unknown>;
  status: string | null;
  onListPipelines: () => void;
  onSavePipeline: (name: string) => void;
  onLoadPipeline: (name: string) => void;
  onDeletePipeline: (name: string) => void;
  onRunPipeline: (input: PipelineRunInput) => void;
  onStopPipeline: () => void;
  onSetGraph: (graph: PipelineGraph) => void;
  onAddNode: (node: PipelineNode) => void;
  onMoveNode: (id: string, position: { x: number; y: number }) => void;
  onRemoveNode: (id: string) => void;
  onAddEdge: (edge: PipelineEdge) => void;
  onRemoveEdge: (id: string) => void;
  onSetEdgeExtract: (
    id: string,
    extract: NonNullable<PipelineEdge["extract"]>,
  ) => void;
}

const DEFAULT_INPUT = JSON.stringify(
  { frames: [[[0]]], encoded: false },
  null,
  2,
);

function newId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Chain saved checkpoints into a DAG and run it as one program. Each node
 * is a saved model; each edge shapes the source node's output into the
 * target's next input via a fixed `extract` mode -- no cycles, no
 * conditional branching (see documentation/model-deployment.md).
 */
export function PipelinePanel({
  models,
  connected,
  graph,
  pipelines,
  running,
  nodeStatus,
  nodeResult,
  status,
  onListPipelines,
  onSavePipeline,
  onLoadPipeline,
  onDeletePipeline,
  onRunPipeline,
  onStopPipeline,
  onSetGraph,
  onAddNode,
  onMoveNode,
  onRemoveNode,
  onAddEdge,
  onRemoveEdge,
  onSetEdgeExtract,
}: Props) {
  const [name, setName] = useState(graph.name || "my-pipeline");
  const [selected, setSelected] = useState("");
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [inputText, setInputText] = useState(DEFAULT_INPUT);
  const [inputError, setInputError] = useState<string | null>(null);

  const nodes: Node[] = useMemo(
    () =>
      (graph.nodes ?? []).map((n) => {
        const result = nodeResult[n.id] as { predicted?: number } | undefined;
        const data: PipelineFlowNodeData = {
          checkpoint: n.checkpoint,
          status: nodeStatus[n.id] ?? "idle",
          predicted: result?.predicted,
        };
        return {
          id: n.id,
          type: "checkpoint",
          position: { x: n.position?.x ?? 0, y: n.position?.y ?? 0 },
          data,
        };
      }),
    [graph.nodes, nodeStatus, nodeResult],
  );

  const edges: Edge[] = useMemo(
    () =>
      (graph.edges ?? []).map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: EXTRACT_LABEL[e.extract ?? "mean_logits"],
        selected: e.id === selectedEdge,
      })),
    [graph.edges, selectedEdge],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      for (const change of changes) {
        if (change.type === "position" && change.position) {
          onMoveNode(change.id, change.position);
        } else if (change.type === "remove") {
          onRemoveNode(change.id);
        }
      }
    },
    [onMoveNode, onRemoveNode],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      for (const change of changes) {
        if (change.type === "remove") {
          onRemoveEdge(change.id);
          if (change.id === selectedEdge) setSelectedEdge(null);
        } else if (change.type === "select" && change.selected) {
          setSelectedEdge(change.id);
        }
      }
    },
    [onRemoveEdge, selectedEdge],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return;
      onAddEdge({
        id: newId("e"),
        source: connection.source,
        target: connection.target,
        extract: "mean_logits",
      });
    },
    [onAddEdge],
  );

  const addSelectedCheckpoint = () => {
    if (!selected) return;
    const count = graph.nodes?.length ?? 0;
    onAddNode({
      id: newId("n"),
      checkpoint: selected,
      position: {
        x: 60 + (count % 4) * 260,
        y: 60 + Math.floor(count / 4) * 160,
      },
    });
  };

  const runWithCurrentInput = () => {
    try {
      const parsed = JSON.parse(inputText) as PipelineRunInput;
      if (!Array.isArray(parsed.frames)) {
        setInputError("input must have a 'frames' array");
        return;
      }
      setInputError(null);
      onRunPipeline(parsed);
    } catch {
      setInputError("input is not valid JSON");
    }
  };

  return (
    <div className="pipeline-panel">
      <div className="pipeline-toolbar panel">
        <div className="control-row">
          <input
            className="text-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="pipeline name…"
            aria-label="Pipeline name"
          />
          <button
            className="apply small"
            onClick={() => name.trim() && onSavePipeline(name.trim())}
            disabled={!name.trim()}
          >
            Save
          </button>
          <button
            className="apply small ghost"
            onClick={() => onSetGraph({ version: 1, name: "", nodes: [], edges: [] })}
          >
            New
          </button>
        </div>

        <div className="control-row">
          <select
            className="text-input"
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
            onClick={() => graph.name && onDeletePipeline(graph.name)}
            disabled={!graph.name}
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
            onClick={addSelectedCheckpoint}
            disabled={!selected}
          >
            Add node
          </button>
        </div>
      </div>

      <div className="pipeline-canvas">
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
          >
            <Background />
            <FlowControls />
            <MiniMap pannable zoomable />
          </ReactFlow>
        </ReactFlowProvider>
      </div>

      <div className="pipeline-side panel">
        {selectedEdge && (
          <div className="pipeline-edge-inspector">
            <div className="panel-title row-title">
              <span>Edge</span>
              <HelpTip text="How the source node's output is shaped into the target node's next input." />
            </div>
            <select
              className="text-input"
              value={
                graph.edges?.find((e) => e.id === selectedEdge)?.extract ??
                "mean_logits"
              }
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
        )}

        <div className="panel-title row-title">
          <span>Run</span>
          <HelpTip text="Feeds every node with no incoming edge. {frames: [...], encoded: true|false} -- the same shape spikeforge-serve reads from stdin." />
        </div>
        <textarea
          className="text-input pipeline-input"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          rows={5}
          aria-label="Pipeline run input (JSON)"
        />
        {inputError && <div className="mismatch">{inputError}</div>}
        <div className="control-row">
          <button
            className="apply"
            onClick={runWithCurrentInput}
            disabled={!connected || running || (graph.nodes?.length ?? 0) === 0}
          >
            {running ? "Running…" : "Run pipeline"}
          </button>
          <button
            className="apply stop small"
            onClick={onStopPipeline}
            disabled={!running}
          >
            Stop
          </button>
        </div>
        {status && <div className="muted">{status}</div>}
      </div>
    </div>
  );
}
