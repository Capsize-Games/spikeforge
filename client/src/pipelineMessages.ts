import { useCallback } from "react";
import type { Dispatch, SetStateAction } from "react";

import type { PipelineGraph } from "./pipelineTypes";
import type { ServerMsg } from "./types";
import type { PipelineState } from "./usePipeline";

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

/** Fold pipeline server messages into the pipeline state setter. */
export function usePipelineMessages(
  setState: Dispatch<SetStateAction<PipelineState>>,
) {
  const handleMessage = useCallback(
    (msg: ServerMsg): boolean => {
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
                nodeStatus[id] =
                  msg.payload.reason === "error" ? "error" : "idle";
              }
            }
            return { ...s, running: false, nodeStatus };
          });
          return true;
        default:
          return false;
      }
    },
    [setState],
  );
  return handleMessage;
}
