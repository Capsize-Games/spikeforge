import { useCallback, useEffect, useRef, useState } from "react";

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

  const send = useCallback((type: string, config: EncodeConfig) => {
    wsRef.current?.send(JSON.stringify({ type, config }));
  }, []);

  const sendTrain = useCallback(
    (type: string, train: TrainConfig, name?: string) => {
      wsRef.current?.send(JSON.stringify({ type, train, name }));
    },
    [],
  );

  const sendNamed = useCallback((type: string, name: string) => {
    wsRef.current?.send(JSON.stringify({ type, name }));
  }, []);

  const sendSelectSample = useCallback((config: EncodeConfig) => {
    wsRef.current?.send(JSON.stringify({ type: "select_sample", config }));
  }, []);

  const sendInfer = useCallback((config: EncodeConfig, train: TrainConfig) => {
    wsRef.current?.send(JSON.stringify({ type: "infer", config, train }));
  }, []);

  const sendStats = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "stats" }));
  }, []);

  return {
    connected,
    send,
    sendTrain,
    sendNamed,
    sendSelectSample,
    sendInfer,
    sendStats,
  };
}
