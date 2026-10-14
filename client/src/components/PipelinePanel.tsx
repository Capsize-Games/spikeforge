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

import { PipelineEdgeInspector } from "./PipelineEdgeInspector";
import {
  PipelineFlowNode,
  type PipelineFlowNodeData,
} from "./PipelineFlowNode";
import { PipelineRunControls } from "./PipelineRunControls";
import { PipelineToolbar } from "./PipelineToolbar";
import { sourceModels } from "./pipelineSources";
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
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);

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

  // A run feeds every node with no incoming edge, so connectivity decides
  // which checkpoints the body has to satisfy -- not position in the array.
  const sources = sourceModels(graph, models);

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

  const addCheckpoint = (checkpoint: string) => {
    const count = graph.nodes?.length ?? 0;
    onAddNode({
      id: newId("n"),
      checkpoint,
      position: {
        x: 60 + (count % 4) * 260,
        y: 60 + Math.floor(count / 4) * 160,
      },
    });
  };

  return (
    <div className="pipeline-panel" data-testid="pipeline-panel">
      <PipelineToolbar
        models={models}
        pipelines={pipelines}
        graphName={graph.name}
        nodeCount={graph.nodes?.length ?? 0}
        edgeCount={graph.edges?.length ?? 0}
        initialName={graph.name || "my-pipeline"}
        onListPipelines={onListPipelines}
        onSavePipeline={onSavePipeline}
        onLoadPipeline={onLoadPipeline}
        onDeletePipeline={onDeletePipeline}
        onSetGraph={onSetGraph}
        onAddCheckpoint={addCheckpoint}
      />

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
          <PipelineEdgeInspector
            selectedEdge={selectedEdge}
            extract={
              graph.edges?.find((e) => e.id === selectedEdge)?.extract ??
              "mean_logits"
            }
            onSetEdgeExtract={onSetEdgeExtract}
          />
        )}

        <PipelineRunControls
          sources={sources}
          connected={connected}
          running={running}
          runnable={(graph.nodes?.length ?? 0) > 0}
          status={status}
          onRun={onRunPipeline}
          onStop={onStopPipeline}
        />
      </div>
    </div>
  );
}
