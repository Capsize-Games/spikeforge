import { useEffect } from "react";
import type { MutableRefObject } from "react";

import { useWebSocket } from "../useWebSocket";
import type { EncodeConfig } from "../types";

type WebSocketControls = ReturnType<typeof useWebSocket>;

/** Seed the server on connect: model list, surrogates, sample, and stats. */
export function useServerBootstrap(
  ws: WebSocketControls,
  configRef: MutableRefObject<EncodeConfig>,
): void {
  useEffect(() => {
    if (!ws.connected) return;
    ws.sendNamed("list_models", "");
    ws.sendAction("surrogates");
    ws.sendAction("targets");
    // Seed the targets panel with the always-available reference report.
    ws.sendNamed("deployment_report", "reference");
    // Populate the viewer immediately so every panel is shown at rest.
    ws.send("configure", configRef.current);
  }, [ws.connected, ws.sendNamed, ws.sendAction, ws.send, configRef]);

  useEffect(() => {
    if (!ws.connected) return;
    ws.sendStats();
    const id = setInterval(ws.sendStats, 2000);
    return () => clearInterval(id);
  }, [ws.connected, ws.sendStats]);
}
