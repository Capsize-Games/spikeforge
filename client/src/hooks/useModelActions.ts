import type { MutableRefObject } from "react";

import { loadCheckpoint } from "../modelLoad";
import { useTraining } from "../useTraining";
import { useWebSocket } from "../useWebSocket";
import type { EncodeConfig } from "../types";

type Training = ReturnType<typeof useTraining>;
type WebSocketControls = ReturnType<typeof useWebSocket>;

interface ActionsOptions {
  config: EncodeConfig;
  configRef: MutableRefObject<EncodeConfig>;
  replaceConfig: (config: EncodeConfig) => void;
  training: Training;
  ws: WebSocketControls;
  setModelLoading: (value: boolean) => void;
  setScrubStep: (value: number | null) => void;
}

/** Model, training, and preview actions used by the dashboard. */
export function useModelActions(options: ActionsOptions) {
  const {
    config,
    configRef,
    replaceConfig,
    training,
    ws,
    setModelLoading,
    setScrubStep,
  } = options;

  /** Data settings are shared, so training mirrors the single config. */
  const sharedPatch = {
    dataset: config.dataset,
    subset: config.subset,
    batch_size: config.batch_size,
    num_steps: config.num_steps,
    encode: config,
  };

  /** Patch the encode config and rebuild the server-side sample. */
  const selectSample = (patch: Partial<EncodeConfig>) => {
    const next = { ...configRef.current, ...patch };
    replaceConfig(next);
    ws.sendSelectSample(next);
  };

  /** Score the currently displayed sample with the loaded model. */
  const runInference = () => {
    ws.sendInfer(configRef.current, training.state.config);
  };

  /** Replay the input spike-frame animation from step 0 (back to live). */
  const playPreview = () => {
    setScrubStep(null);
    ws.send("run", configRef.current);
  };

  const stopPreview = () => ws.send("stop", configRef.current);

  const train = () => {
    const merged = { ...training.state.config, ...sharedPatch };
    training.patch(sharedPatch);
    training.reset();
    training.setRunning(true);
    ws.sendTrain("train", merged);
  };

  const stopTrain = () => {
    training.setRunning(false);
    ws.sendTrain("stop_train", training.state.config);
  };

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

  const saveModel = (name: string) => ws.sendNamed("save_model", name);
  const newModel = () => ws.sendNamed("new_model", "");

  /** Apply a checkpoint's saved architecture/encoding, then load it. */
  const loadModel = (name: string) => {
    setModelLoading(true);
    const merged = loadCheckpoint(name, {
      saved: training.state.models.find((m) => m.name === name),
      current: configRef.current,
      config: training.state.config,
      sharedPatch,
      replaceConfig,
      applyPatch: training.patch,
    });
    ws.sendTrain("load_model", merged, name);
  };

  return {
    selectSample,
    runInference,
    playPreview,
    stopPreview,
    train,
    stopTrain,
    saveModel,
    loadModel,
    newModel,
    requestTrajectory,
    requestMetrics,
    requestEncodingReport,
    requestSurrogateCurve,
    requestBenchmark,
    requestNirExport,
    requestNirValidate,
    requestTargets,
    requestDeploymentReport,
  };
}
