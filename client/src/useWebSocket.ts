import { useCallback, useEffect, useRef, useState } from "react";

import type { EncodeConfig, ServerMsg } from "./types";

interface Options {
  onMessage: (msg: ServerMsg) => void;
}

export function useWebSocket({ onMessage }: Options) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      onMessage(JSON.parse(event.data as string) as ServerMsg);
    };

    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const send = useCallback((type: string, config: EncodeConfig) => {
    wsRef.current?.send(JSON.stringify({ type, config }));
  }, []);

  return { connected, send };
}
