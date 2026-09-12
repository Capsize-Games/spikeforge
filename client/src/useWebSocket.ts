import { useCallback, useEffect, useRef, useState } from "react";

import type { HubQueryInput } from "./hubTypes";
import type { PipelineGraph, PipelineRunInput } from "./pipelineTypes";
import { PROTOCOL_VERSION } from "./protocol/generated";
import type { EncodeConfig, ServerMsg, TrainConfig } from "./types";

interface Options {
  onMessage: (msg: ServerMsg) => void;
}

const RETRY_MS = 1500;

export function useWebSocket({ onMessage }: Options) {
  const wsRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  const closedRef = useRef(false);
  const [connected, setConnected] = useState(false);

  // Keep the newest handler without re-opening the socket.
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let timer: number | undefined;
    const connect = () => {
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${protocol}://${location.host}/ws`);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onmessage = (event) =>
        onMessageRef.current(JSON.parse(event.data as string) as ServerMsg);
      ws.onerror = () => ws.close();
      ws.onclose = () => {
        setConnected(false);
        if (!closedRef.current) {
          timer = window.setTimeout(connect, RETRY_MS);
        }
      };
    };
    connect();
    return () => {
      closedRef.current = true;
      if (timer) window.clearTimeout(timer);
      wsRef.current?.close();
    };
  }, []);

  /**
   * Send one JSON message, stamping the protocol version every outbound
   * message requires (server/app.py rejects anything missing it).
   */
  const sendJson = useCallback((message: Record<string, unknown>) => {
    wsRef.current?.send(
      JSON.stringify({ protocol_version: PROTOCOL_VERSION, ...message }),
    );
  }, []);

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
    connected,
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
