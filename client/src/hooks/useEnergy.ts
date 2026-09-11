import { useCallback, useState } from "react";

import type { EnergyPayload } from "../energyTypes";
import type { ServerMsg, TrainConfig } from "../types";
import { useWebSocket } from "../useWebSocket";

type WebSocketControls = ReturnType<typeof useWebSocket>;

interface Options {
  ws: WebSocketControls;
  connected: boolean;
}

/** Own the event-driven energy report and request it for a target. */
export function useEnergy({ ws, connected }: Options) {
  const { sendTrain } = ws;
  const [payload, setPayload] = useState<EnergyPayload | null>(null);
  const [loading, setLoading] = useState(false);

  /** Fold one server message into the energy state; other types pass. */
  const onMessage = useCallback((msg: ServerMsg): boolean => {
    switch (msg.type) {
      case "energy_report":
        setPayload(msg.payload);
        setLoading(false);
        return true;
      case "error":
        // A failed request must not leave the estimate spinner stuck; the
        // viewer owns the visible error banner, so return false.
        setLoading(false);
        return false;
      default:
        return false;
    }
  }, []);

  /** Request the report for `target` using the current training config. */
  const run = useCallback(
    (config: TrainConfig, target: string) => {
      setLoading(true);
      sendTrain("energy_report", config, target);
    },
    [sendTrain],
  );

  return { payload, loading, connected, onMessage, run };
}
