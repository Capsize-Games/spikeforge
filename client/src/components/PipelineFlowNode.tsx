import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";

import type { PipelineNodeStatus } from "../pipelineTypes";

export interface PipelineFlowNodeData extends Record<string, unknown> {
  checkpoint: string;
  status: PipelineNodeStatus;
  predicted?: number;
}

const STATUS_LABEL: Record<PipelineNodeStatus, string> = {
  idle: "idle",
  running: "running…",
  done: "done",
  error: "error",
};

/** One pipeline node: a checkpoint name, its live status, and connectors. */
export function PipelineFlowNode({ data, selected }: NodeProps) {
  const nodeData = data as PipelineFlowNodeData;
  return (
    <div className={`pipeline-node status-${nodeData.status} ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <div className="pipeline-node-name">{nodeData.checkpoint}</div>
      <div className="pipeline-node-status">
        <span className={`pipeline-node-dot status-${nodeData.status}`} />
        {STATUS_LABEL[nodeData.status]}
        {nodeData.status === "done" && nodeData.predicted !== undefined && (
          <span className="pipeline-node-predicted">
            → class {nodeData.predicted}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
