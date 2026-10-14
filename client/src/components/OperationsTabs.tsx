import { lazy, Suspense } from "react";

import { EnergyPanel } from "./EnergyPanel";
import { HubPanel } from "./HubPanel";
import { TabPanel } from "./TabPanel";
import { TargetsPanel } from "./TargetsPanel";
import { useEnergy } from "../hooks/useEnergy";
import { useHub } from "../hooks/useHub";
import { useModelActions } from "../hooks/useModelActions";
import { useViewer } from "../hooks/useViewer";
import { useI18n } from "../i18n/I18nProvider";
import type { TabId } from "../tabs";
import { usePipeline } from "../usePipeline";
import { useTraining } from "../useTraining";
import { useWebSocket } from "../useWebSocket";

type Training = ReturnType<typeof useTraining>;
type Viewer = ReturnType<typeof useViewer>;
type Actions = ReturnType<typeof useModelActions>;
type WebSocketControls = ReturnType<typeof useWebSocket>;
type Hub = ReturnType<typeof useHub>;
type Energy = ReturnType<typeof useEnergy>;
type Pipeline = ReturnType<typeof usePipeline>;

// React Flow is a real bundle addition (~100kb), so the Pipeline tab is the
// app's first code-split boundary -- everyone else's initial load stays
// exactly as it was.
const PipelinePanel = lazy(() =>
  import("./PipelinePanel").then((m) => ({
    default: m.PipelinePanel,
  })),
);

interface Props {
  active: TabId;
  ws: WebSocketControls;
  hub: Hub;
  energy: Energy;
  pipeline: Pipeline;
  training: Training;
  viewer: Viewer;
  actions: Actions;
}

/** The hub, deploy, and pipeline tabs of the dashboard. */
export function OperationsTabs({
  active,
  ws,
  hub,
  energy,
  pipeline,
  training,
  viewer,
  actions,
}: Props) {
  const { t } = useI18n();

  return (
    <>
      <TabPanel id="hub" active={active}>
        <div className="tab-col hub-col">
          <HubPanel hub={hub} />
        </div>
      </TabPanel>

      <TabPanel id="deploy" active={active}>
        <div className="tab-cols tab-cols-deploy">
          <div className="tab-col">
            <TargetsPanel
              list={viewer.state.targetList}
              report={viewer.state.deploymentReport}
              backendRun={viewer.state.backendRun}
              onRefresh={actions.requestTargets}
              onSelectTarget={actions.requestDeploymentReport}
              onRunBackend={actions.requestDeployRun}
            />
          </div>

          <div className="tab-col">
            <EnergyPanel
              payload={energy.payload}
              targets={viewer.state.targetList?.targets ?? []}
              loading={energy.loading}
              onRun={(target) => energy.run(training.state.config, target)}
            />
          </div>
        </div>
      </TabPanel>

      <TabPanel id="pipeline" active={active}>
        <Suspense
          fallback={<div className="muted">{t("pipeline.loading")}</div>}
        >
          <PipelinePanel
            models={training.state.models}
            connected={ws.connected}
            graph={pipeline.state.graph}
            pipelines={pipeline.state.pipelines}
            running={pipeline.state.running}
            nodeStatus={pipeline.state.nodeStatus}
            nodeResult={pipeline.state.nodeResult}
            status={pipeline.state.status}
            onListPipelines={() => ws.sendAction("list_pipelines")}
            onSavePipeline={(name) =>
              ws.sendSavePipeline(name, pipeline.state.graph)
            }
            onLoadPipeline={(name) =>
              ws.sendPipelineName("load_pipeline", name)
            }
            onDeletePipeline={(name) =>
              ws.sendPipelineName("delete_pipeline", name)
            }
            onRunPipeline={(input) =>
              ws.sendRunPipeline(pipeline.state.graph, input)
            }
            onStopPipeline={() => ws.sendAction("stop_pipeline")}
            onSetGraph={pipeline.setGraph}
            onAddNode={pipeline.addNode}
            onMoveNode={pipeline.moveNode}
            onRemoveNode={pipeline.removeNode}
            onAddEdge={pipeline.addEdge}
            onRemoveEdge={pipeline.removeEdge}
            onSetEdgeExtract={pipeline.setEdgeExtract}
          />
        </Suspense>
      </TabPanel>
    </>
  );
}
