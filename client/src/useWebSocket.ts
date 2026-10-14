import { useCallback, useEffect, useRef, useState } from "react";

import { accessToken } from "./accessToken";
import { PROTOCOL_VERSION } from "./protocol/generated";
import type { ServerMsg } from "./types";
import { useSenders } from "./websocketSenders";

interface Options {
  onMessage: (msg: ServerMsg) => void;
}

const RETRY_MS = 1500;

export function useWebSocket({ onMessage }: Options) {
  const wsRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  const closedRef = useRef(false);
  const [connected, setConnected] = useState(false);
  const [unauthorized, setUnauthorized] = useState(false);

  // Keep the newest handler without re-opening the socket.
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let timer: number | undefined;
    const connect = () => {
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const token = accessToken();
      const query = token ? `?token=${encodeURIComponent(token)}` : "";
      const ws = new WebSocket(
        `${protocol}://${location.host}/ws${query}`,
      );
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        setUnauthorized(false);
      };
      ws.onmessage = (event) =>
        onMessageRef.current(JSON.parse(event.data as string) as ServerMsg);
      ws.onerror = () => ws.close();
      ws.onclose = (event) => {
        setConnected(false);
        // 1008 (policy violation) is what the server closes with when
        // SPIKEFORGE_DASHBOARD_TOKEN is set and the token is missing or
        // wrong -- retrying with the same bad token would just spin.
        if (event.code === 1008) {
          setUnauthorized(true);
          return;
        }
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

  const senders = useSenders(sendJson);

  return {
    connected,
    unauthorized,
    ...senders,
  };
}
