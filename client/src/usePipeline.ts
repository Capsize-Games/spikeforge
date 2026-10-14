import { useCallback, useState } from "react";

import { usePipelineMessages } from "./pipelineMessages";
import type {
  PipelineGraph,
  PipelineNodeStatus,
  PipelineRunInput,
  PipelineSummary,
} from "./pipelineTypes";

type PipelineNode = NonNullable<PipelineGraph["nodes"]>[number];
type PipelineEdge = NonNullable<PipelineGraph["edges"]>[number];

export interface PipelineState {
  graph: PipelineGraph;
  pipelines: PipelineSummary[];
  running: boolean;
  /** Per-node status, keyed by node id; absent entries read as "idle". */
  nodeStatus: Record<string, PipelineNodeStatus>;
  /** Per-node last result, keyed by node id, kept across runs until edited. */
  nodeResult: Record<string, unknown>;
  status: string | null;
}

const EMPTY_GRAPH: PipelineGraph = {
  version: 1,
  name: "",
  nodes: [],
  edges: [],
};

const initial: PipelineState = {
  graph: EMPTY_GRAPH,
  pipelines: [],
  running: false,
  nodeStatus: {},
  nodeResult: {},
  status: null,
};

/** Track the pipeline graph being edited, saved pipelines, and run state. */
export function usePipeline() {
  const [state, setState] = useState<PipelineState>(initial);

  const handleMessage = usePipelineMessages(setState);

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
