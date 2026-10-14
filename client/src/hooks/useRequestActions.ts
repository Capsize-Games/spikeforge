import { useTraining } from "../useTraining";
import { useWebSocket } from "../useWebSocket";

type Training = ReturnType<typeof useTraining>;
type WebSocketControls = ReturnType<typeof useWebSocket>;

interface RequestsOptions {
  training: Training;
  ws: WebSocketControls;
}

/** One-shot server requests: reports, curves, targets, and NIR exports. */
export function useRequestActions({ training, ws }: RequestsOptions) {
  /** Capture bounded U[t]/I[t]/S[t] traces (educational mode only). */
  const requestTrajectory = () => ws.sendAction("trajectory");

  /** Capture aggregate firing-rate/sparsity/ISI metrics (educational only). */
  const requestMetrics = () => ws.sendAction("metrics");

  /** Decode the configured sample into an encoding report. */
  const requestEncodingReport = () => ws.sendAction("encoding_report");

  /** Sample the named surrogate gradient's derivative curve. */
  const requestSurrogateCurve = (name: string) =>
    ws.sendNamed("surrogate_curve", name);

  /** Run the tiny benchmark fixture from the current training config. */
  const requestBenchmark = () => {
    training.setBenchmarkLoading(true);
    ws.sendTrain("benchmark", training.state.config);
  };

  /** Export the active/configured topology as a NIR graph summary. */
  const requestNirExport = () =>
    ws.sendTrain("nir_export", training.state.config);

  /** Run the independent NIR interpreter and report drift. */
  const requestNirValidate = () =>
    ws.sendTrain("nir_validate", training.state.config);

  /** Fetch the availability-annotated deployment target registry. */
  const requestTargets = () => ws.sendAction("targets");

  /** Request the deployment report for the named target. */
  const requestDeploymentReport = (target: string) =>
    ws.sendNamed("deployment_report", target);

  /** Compile and run the configured topology on the named backend. */
  const requestDeployRun = (target: string) =>
    ws.sendTrain("deploy_run", training.state.config, target);

  return {
    requestTrajectory,
    requestMetrics,
    requestEncodingReport,
    requestSurrogateCurve,
    requestBenchmark,
    requestNirExport,
    requestNirValidate,
    requestTargets,
    requestDeploymentReport,
    requestDeployRun,
  };
}
