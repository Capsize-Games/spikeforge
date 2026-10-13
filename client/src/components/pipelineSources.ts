/**
 * Which nodes a pipeline run actually feeds.
 *
 * `run_pipeline` hands the request body to every node with **no incoming
 * edge**. Those are the nodes whose input contract the run body has to match.
 * Position in `graph.nodes` says nothing about that: the interface lets a user
 * add nodes in any order and connect them afterwards, so the first entry in
 * the array is frequently a downstream node.
 */

import type { PipelineGraph } from "../pipelineTypes";
import type { SavedModel } from "../types";

type GraphNode = NonNullable<PipelineGraph["nodes"]>[number];

/**
 * Return the nodes with no incoming edge, in graph order.
 *
 * A graph with nodes but no edges has every node as a source, which is
 * correct: the runner feeds all of them.
 */
export function sourceNodes(graph: PipelineGraph): GraphNode[] {
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];
  return nodes.filter(
    (node) => !edges.some((edge) => edge.target === node.id),
  );
}

/** The saved checkpoints the run body will be fed into. */
export function sourceModels(
  graph: PipelineGraph,
  models: SavedModel[],
): SavedModel[] {
  const found: SavedModel[] = [];
  for (const node of sourceNodes(graph)) {
    const model = models.find(
      (candidate) => candidate.name === node.checkpoint,
    );
    if (model !== undefined) found.push(model);
  }
  return found;
}
