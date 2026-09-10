import type { NirGraphNode, NirGraphPayload } from "../nirTypes";

/** Fixed node box and spacing; the SVG viewBox scales to the panel. */
export const NODE_W = 132;
export const NODE_H = 38;
const COL_GAP = 48;
const ROW_GAP = 12;
const PAD = 12;

export interface PlacedNode {
  name: string;
  kind: string;
  x: number;
  y: number;
}

export interface PlacedEdge {
  key: string;
  path: string;
  delayed: boolean;
}

export interface GraphLayout {
  nodes: PlacedNode[];
  edges: PlacedEdge[];
  width: number;
  height: number;
}

/**
 * Longest-path layering: each node sits one column right of its latest
 * predecessor. Cycles (recurrent edges) are bounded by the node count, so a
 * skip/multi-branch/recurrent graph still lays out instead of looping.
 */
function assignLayers(graph: NirGraphPayload): Map<string, number> {
  const layer = new Map<string, number>();
  for (const node of graph.nodes) layer.set(node.name, 0);
  for (let pass = 0; pass < graph.nodes.length; pass += 1) {
    let changed = false;
    for (const edge of graph.edges) {
      if (!layer.has(edge.source) || !layer.has(edge.target)) continue;
      const next = (layer.get(edge.source) ?? 0) + 1;
      if ((layer.get(edge.target) ?? 0) < next) {
        layer.set(edge.target, next);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return layer;
}

/** Group nodes by depth, preserving their original order within a depth. */
function groupByLayer(
  nodes: NirGraphNode[],
  layer: Map<string, number>,
): Map<number, NirGraphNode[]> {
  const groups = new Map<number, NirGraphNode[]>();
  for (const node of nodes) {
    const bucket = groups.get(layer.get(node.name) ?? 0);
    if (bucket) bucket.push(node);
    else groups.set(layer.get(node.name) ?? 0, [node]);
  }
  return groups;
}

/** Place each node in a column/row grid and report the grid bounds. */
function placeNodes(groups: Map<number, NirGraphNode[]>): {
  nodes: PlacedNode[];
  columns: number;
  rows: number;
} {
  const depths = [...groups.keys()].sort((a, b) => a - b);
  const placed: PlacedNode[] = [];
  let rows = 0;
  depths.forEach((depth, col) => {
    const bucket = groups.get(depth) ?? [];
    rows = Math.max(rows, bucket.length);
    bucket.forEach((node, row) => {
      placed.push({
        name: node.name,
        kind: node.kind,
        x: PAD + col * (NODE_W + COL_GAP),
        y: PAD + row * (NODE_H + ROW_GAP),
      });
    });
  });
  return { nodes: placed, columns: Math.max(1, depths.length), rows };
}

/** Straight path for forward edges; a bowed arc for skip/back/self edges. */
function edgePath(source: PlacedNode, target: PlacedNode): string {
  const sx = source.x + NODE_W;
  const sy = source.y + NODE_H / 2;
  const tx = target.x;
  const ty = target.y + NODE_H / 2;
  const forward = tx > sx + 1 && source.name !== target.name;
  if (forward) {
    return `M ${sx} ${sy} L ${tx} ${ty}`;
  }
  const bow = Math.max(24, Math.abs(ty - sy) * 0.5 + 20);
  return (
    `M ${sx} ${sy} C ${sx + 40} ${sy + bow}, ` +
    `${tx - 40} ${ty + bow}, ${tx} ${ty}`
  );
}

/** Lay out a NIR graph summary with no external layout dependency. */
export function layoutGraph(graph: NirGraphPayload): GraphLayout {
  const placed = placeNodes(groupByLayer(graph.nodes, assignLayers(graph)));
  const byName = new Map(placed.nodes.map((node) => [node.name, node]));
  const edges: PlacedEdge[] = [];
  graph.edges.forEach((edge, index) => {
    const source = byName.get(edge.source);
    const target = byName.get(edge.target);
    if (!source || !target) return;
    edges.push({
      key: `${edge.source}->${edge.target}#${index}`,
      path: edgePath(source, target),
      delayed: edge.delayed,
    });
  });
  return {
    nodes: placed.nodes,
    edges,
    width:
      PAD * 2 + placed.columns * NODE_W + (placed.columns - 1) * COL_GAP,
    height: PAD * 2 + placed.rows * NODE_H +
      Math.max(0, placed.rows - 1) * ROW_GAP,
  };
}
