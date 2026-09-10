import { useCallback, useEffect, useReducer, useState } from "react";
import type { MutableRefObject } from "react";

import { readBool, STORAGE_KEYS, writeBool } from "../storage";
import { useTraining } from "../useTraining";
import type { EncodeConfig, ServerMsg, SystemStatsPayload } from "../types";
import type { TrainConfig } from "../types";
import { createInitialViewerState, reduceViewer } from "../viewerState";

type Training = ReturnType<typeof useTraining>;

interface ViewerOptions {
  config: EncodeConfig;
  configRef: MutableRefObject<EncodeConfig>;
  training: Training;
  sendInfer: (config: EncodeConfig, train: TrainConfig) => void;
  connected: boolean;
}

/** Track streamed viewer state, stats, and derived display values. */
export function useViewer(options: ViewerOptions) {
  const { config, configRef, training, sendInfer, connected } = options;
  const [state, dispatch] = useReducer(
    reduceViewer,
    readBool(STORAGE_KEYS.autoPredict, false),
    createInitialViewerState,
  );
  const [stats, setStats] = useState<SystemStatsPayload | null>(null);
  const [scrubStep, setScrubStep] = useState<number | null>(null);

  useEffect(() => {
    writeBool(STORAGE_KEYS.autoPredict, state.autoPredict);
  }, [state.autoPredict]);

  const onMessage = useCallback(
    (msg: ServerMsg) => {
      if (
        msg.type === "model_loaded" ||
        msg.type === "model_cleared" ||
        msg.type === "error"
      ) {
        dispatch({ kind: "modelLoading", value: false });
      }
      if (msg.type === "system_stats") {
        setStats(msg.payload);
      }
      const trained =
        msg.type === "train_state" &&
        msg.payload.running === false &&
        msg.payload.reason === "finished";
      // Refresh layer-activity rasters when a model appears; the prediction
      // display itself stays gated by the auto-predict toggle.
      if (msg.type === "model_loaded" || trained) {
        window.setTimeout(
          () => sendInfer(configRef.current, training.state.config),
          0,
        );
      }
      if (training.handleMessage(msg)) return;
      dispatch({ kind: "message", msg });
    },
    [configRef, sendInfer, training],
  );

  const setModelLoading = useCallback(
    (value: boolean) => dispatch({ kind: "modelLoading", value }),
    [],
  );
  const setAutoPredict = useCallback(
    (value: boolean) => dispatch({ kind: "autoPredict", value }),
    [],
  );
  const clearInference = useCallback(
    () => dispatch({ kind: "clearInference" }),
    [],
  );

  const gpuAvailable = stats ? stats.device.available.includes("gpu") : true;
  const numSteps =
    state.rasters.input?.num_steps ??
    state.status?.num_steps ??
    config.num_steps;
  const timeStep = scrubStep ?? (state.running ? state.spikeStep : null);
  const displayFrame =
    (scrubStep !== null ? state.framesByStep[scrubStep] : undefined) ??
    state.spikeFrames.input;
  const compatibility = training.state.loaded?.compatibility;
  const hasModel =
    training.state.loaded !== null || training.state.last !== null;
  const canAutoPredict =
    connected && hasModel && (compatibility?.dataset_match ?? true);

  return {
    state,
    stats,
    setStats,
    scrubStep,
    setScrubStep,
    autoPredict: state.autoPredict,
    setAutoPredict,
    modelLoading: state.modelLoading,
    setModelLoading,
    clearInference,
    onMessage,
    gpuAvailable,
    numSteps,
    timeStep,
    displayFrame,
    compatibility,
    hasModel,
    canAutoPredict,
    locked: training.state.loaded !== null,
  };
}
