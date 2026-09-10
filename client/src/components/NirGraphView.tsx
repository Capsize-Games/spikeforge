import { useMemo } from "react";

import type { NirGraphPayload } from "../nirTypes";
import { layoutGraph, NODE_H, NODE_W } from "./graphLayout";

interface Props {
  graph: NirGraphPayload;
}

/** Truncate long node names so they fit inside the fixed node box. */
function shortName(name: string): string {
  return name.length > 18 ? `${name.slice(0, 17)}…` : name;
}

/**
 * Dependency-free layered rendering of a NIR graph summary. Columns come from
 * longest-path layering, so skip, multi-branch, and recurrent graphs render
 * rather than assuming a linear chain. Delayed edges are dashed.
 */
export function NirGraphView({ graph }: Props) {
  const layout = useMemo(() => layoutGraph(graph), [graph]);
  return (
    <svg
      className="nir-graph"
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      role="img"
      aria-label="NIR topology graph"
    >
      <defs>
        <marker
          id="nir-arrow"
          markerWidth="8"
          markerHeight="8"
          refX="7"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L7,3 L0,6 Z" className="nir-arrow-head" />
        </marker>
        <marker
          id="nir-arrow-delayed"
          markerWidth="8"
          markerHeight="8"
          refX="7"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L7,3 L0,6 Z" className="nir-arrow-head delayed" />
        </marker>
      </defs>

      {layout.edges.map((edge) => (
        <path
          key={edge.key}
          className={edge.delayed ? "nir-edge delayed" : "nir-edge"}
          d={edge.path}
          markerEnd={
            edge.delayed ? "url(#nir-arrow-delayed)" : "url(#nir-arrow)"
          }
        />
      ))}

      {layout.nodes.map((node) => (
        <g key={node.name}>
          <rect
            className="nir-node"
            x={node.x}
            y={node.y}
            width={NODE_W}
            height={NODE_H}
            rx={4}
          />
          <text className="nir-node-name" x={node.x + 8} y={node.y + 15}>
            {shortName(node.name)}
          </text>
          <text className="nir-node-kind" x={node.x + 8} y={node.y + 30}>
            {node.kind}
          </text>
        </g>
      ))}
    </svg>
  );
}
