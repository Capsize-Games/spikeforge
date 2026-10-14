import { useCallback } from "react";

import type { HubQueryInput } from "./hubTypes";
import type { PipelineGraph, PipelineRunInput } from "./pipelineTypes";
import type { EncodeConfig, TrainConfig } from "./types";

/** Outbound JSON sender that stamps the protocol version. */
export type SendJson = (message: Record<string, unknown>) => void;

/** Build the typed per-action send helpers over one JSON sender. */
export function useSenders(sendJson: SendJson) {
  const send = useCallback(
    (type: string, config: EncodeConfig) => {
      sendJson({ type, config });
    },
    [sendJson],
  );

  const sendTrain = useCallback(
    (type: string, train: TrainConfig, name?: string) => {
      sendJson({ type, train, name });
    },
    [sendJson],
  );

  const sendNamed = useCallback(
    (type: string, name: string) => {
      sendJson({ type, name });
    },
    [sendJson],
  );

  /** Send a parameterless action (e.g. "surrogates", "metrics"). */
  const sendAction = useCallback(
    (type: string) => {
      sendJson({ type });
    },
    [sendJson],
  );

  const sendSelectSample = useCallback(
    (config: EncodeConfig) => {
      sendJson({ type: "select_sample", config });
    },
    [sendJson],
  );

  const sendInfer = useCallback(
    (config: EncodeConfig, train: TrainConfig) => {
      sendJson({ type: "infer", config, train });
    },
    [sendJson],
  );

  const sendStats = useCallback(() => {
    sendJson({ type: "stats" });
  }, [sendJson]);

  const sendCancelDownload = useCallback(() => {
    sendJson({ type: "cancel_download" });
  }, [sendJson]);

  /** Send a hub action with its additive query fields and optional name. */
  const sendHub = useCallback(
    (type: string, hub: HubQueryInput, name?: string) => {
      sendJson({ type, hub, name });
    },
    [sendJson],
  );

  /** list_pipelines / load_pipeline / delete_pipeline: act on a saved name. */
  const sendPipelineName = useCallback(
    (type: string, name?: string) => {
      sendJson({ type, name });
    },
    [sendJson],
  );

  /** save_pipeline: persist the current graph under `name`. */
  const sendSavePipeline = useCallback(
    (name: string, pipeline: PipelineGraph) => {
      sendJson({ type: "save_pipeline", name, pipeline });
    },
    [sendJson],
  );

  /** run_pipeline: run the current graph against a raw input request. */
  const sendRunPipeline = useCallback(
    (pipeline: PipelineGraph, pipelineInput: PipelineRunInput) => {
      sendJson({
        type: "run_pipeline",
        pipeline,
        pipeline_input: pipelineInput,
      });
    },
    [sendJson],
  );

  return {
    send,
    sendTrain,
    sendNamed,
    sendAction,
    sendSelectSample,
    sendInfer,
    sendStats,
    sendCancelDownload,
    sendHub,
    sendPipelineName,
    sendSavePipeline,
    sendRunPipeline,
  };
}
