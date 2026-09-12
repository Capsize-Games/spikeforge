import { useCallback, useState } from "react";

import type {
  PipelineGraph,
  PipelineNodeStatus,
  PipelineRunInput,
  PipelineSummary,
} from "./pipelineTypes";
import type { ServerMsg } from "./types";

type PipelineNode = NonNullable<PipelineGraph["nodes"]>[number];
type PipelineEdge = NonNullable<PipelineGraph["edges"]>[number];

interface PipelineState {
  graph: PipelineGraph;
  pipelines: PipelineSummary[];
  running: boolean;
  /** Per-node status, keyed by node id; absent entries read as "idle". */
  nodeStatus: Record<string, PipelineNodeStatus>;
  /** Per-node last result, keyed by node id, kept across runs until edited. */
  nodeResult: Record<string, unknown>;
  status: string | null;
}

const EMPTY_GRAPH: PipelineGraph = { version: 1, name: "", nodes: [], edges: [] };

const initial: PipelineState = {
  graph: EMPTY_GRAPH,
  pipelines: [],
  running: false,
  nodeStatus: {},
  nodeResult: {},
  status: null,
};

/** Return the node ids in the order edges say they must run, best-effort.
 *
 * Mirrors spikeforge_serve.pipeline.PipelineGraph.topological_order's Kahn's
 * algorithm, but tolerates a cycle (returns whatever it could order) since
 * this is only used to guess which node is "running" next for the UI, not
 * to decide correctness -- the server is the real authority on that.
 */
function bestEffortOrder(graph: PipelineGraph): string[] {
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];
  const inDegree = new Map<string, number>(nodes.map((n) => [n.id, 0]));
  for (const edge of edges) {
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
  }
  const ready = [...inDegree.entries()]
    .filter(([, degree]) => degree === 0)
    .map(([id]) => id)
    .sort();
  const ordered: string[] = [];
  const remaining = [...edges];
  while (ready.length > 0) {
    const id = ready.shift() as string;
    ordered.push(id);
    const outgoing = remaining.filter((e) => e.source === id);
    for (const edge of outgoing) {
      remaining.splice(remaining.indexOf(edge), 1);
      const next = (inDegree.get(edge.target) ?? 1) - 1;
      inDegree.set(edge.target, next);
      if (next === 0) ready.push(edge.target);
    }
    ready.sort();
  }
  return ordered;
}

/** Track the pipeline graph being edited, saved pipelines, and run state. */
export function usePipeline() {
  const [state, setState] = useState<PipelineState>(initial);

  const handleMessage = useCallback((msg: ServerMsg): boolean => {
    switch (msg.type) {
      case "pipeline_list":
        setState((s) => ({ ...s, pipelines: msg.payload.pipelines }));
        return true;
      case "pipeline_saved":
        setState((s) => ({
          ...s,
          status: `saved ${msg.payload.name}`,
          graph: { ...s.graph, name: msg.payload.name },
        }));
        return true;
      case "pipeline_loaded":
        setState((s) => ({
          ...s,
          graph: msg.payload,
          nodeStatus: {},
          nodeResult: {},
        }));
        return true;
      case "pipeline_deleted":
        setState((s) => ({
          ...s,
          status: msg.payload.existed
            ? `deleted ${msg.payload.name}`
            : `${msg.payload.name} was already gone`,
        }));
        return true;
      case "pipeline_node_result":
        setState((s) => {
          const order = bestEffortOrder(s.graph);
          const doneIndex = order.indexOf(msg.payload.node_id);
          const next = order[doneIndex + 1];
          const nodeStatus = { ...s.nodeStatus };
          nodeStatus[msg.payload.node_id] = "done";
          if (next !== undefined) nodeStatus[next] = "running";
          return {
            ...s,
            nodeStatus,
            nodeResult: {
              ...s.nodeResult,
              [msg.payload.node_id]: msg.payload.result,
            },
          };
        });
        return true;
      case "pipeline_run_state":
        setState((s) => {
          if (msg.payload.running) {
            const order = bestEffortOrder(s.graph);
            const first = order[0];
            return {
              ...s,
              running: true,
              nodeStatus: first === undefined ? {} : { [first]: "running" },
              nodeResult: {},
            };
          }
          // A run just ended: anything still "running" didn't finish --
          // mark it (and only it) as the error site when the run failed.
          const nodeStatus = { ...s.nodeStatus };
          for (const [id, status] of Object.entries(nodeStatus)) {
            if (status === "running") {
              nodeStatus[id] = msg.payload.reason === "error" ? "error" : "idle";
            }
          }
          return { ...s, running: false, nodeStatus };
        });
        return true;
      default:
        return false;
    }
  }, []);

  const setGraph = useCallback((graph: PipelineGraph) => {
    setState((s) => ({ ...s, graph, nodeStatus: {}, nodeResult: {} }));
  }, []);

  const addNode = useCallback((node: PipelineNode) => {
    setState((s) => ({
      ...s,
      graph: { ...s.graph, nodes: [...(s.graph.nodes ?? []), node] },
    }));
  }, []);

  const moveNode = useCallback(
    (id: string, position: { x: number; y: number }) => {
      setState((s) => ({
        ...s,
        graph: {
          ...s.graph,
          nodes: (s.graph.nodes ?? []).map((n) =>
            n.id === id ? { ...n, position } : n,
          ),
        },
      }));
    },
    [],
  );

  const removeNode = useCallback((id: string) => {
    setState((s) => ({
      ...s,
      graph: {
        ...s.graph,
        nodes: (s.graph.nodes ?? []).filter((n) => n.id !== id),
        edges: (s.graph.edges ?? []).filter(
          (e) => e.source !== id && e.target !== id,
        ),
      },
    }));
  }, []);

  const addEdge = useCallback((edge: PipelineEdge) => {
    setState((s) => ({
      ...s,
      graph: { ...s.graph, edges: [...(s.graph.edges ?? []), edge] },
    }));
  }, []);

  const removeEdge = useCallback((id: string) => {
    setState((s) => ({
      ...s,
      graph: {
        ...s.graph,
        edges: (s.graph.edges ?? []).filter((e) => e.id !== id),
      },
    }));
  }, []);

  const setEdgeExtract = useCallback(
    (id: string, extract: NonNullable<PipelineEdge["extract"]>) => {
      setState((s) => ({
        ...s,
        graph: {
          ...s.graph,
          edges: (s.graph.edges ?? []).map((e) =>
            e.id === id ? { ...e, extract } : e,
          ),
        },
      }));
    },
    [],
  );

  return {
    state,
    handleMessage,
    setGraph,
    addNode,
    moveNode,
    removeNode,
    addEdge,
    removeEdge,
    setEdgeExtract,
  };
}

export type { PipelineNode, PipelineEdge, PipelineRunInput };
