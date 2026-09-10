import { useCallback, useState } from "react";

import { Controls } from "./components/Controls";
import { HeatmapCanvas } from "./components/HeatmapCanvas";
import { RasterCanvas } from "./components/RasterCanvas";
import type {
  EncodeConfig,
  RasterPayload,
  ServerMsg,
  StatusPayload,
} from "./types";
import { defaultConfig } from "./types";
import { useWebSocket } from "./useWebSocket";

interface ViewerState {
  sample: number[][] | null;
  spikeFrame: number[][] | null;
  reconGain1: number[][] | null;
  reconLow: number[][] | null;
  raster: RasterPayload | null;
  status: StatusPayload | null;
  error: string | null;
  running: boolean;
}

const initial: ViewerState = {
  sample: null,
  spikeFrame: null,
  reconGain1: null,
  reconLow: null,
  raster: null,
  status: null,
  error: null,
  running: false,
};

export default function App() {
  const [config, setConfig] = useState<EncodeConfig>(defaultConfig);
  const [state, setState] = useState<ViewerState>(initial);

  const handleMessage = useCallback((msg: ServerMsg) => {
    switch (msg.type) {
      case "config_ack":
        setState((s) => ({
          ...s,
          error: null,
          sample: null,
          spikeFrame: null,
          reconGain1: null,
          reconLow: null,
        }));
        break;
      case "image":
        {
          const kind = (msg as { kind?: string }).kind;
          if (kind === "recon_gain1") {
            setState((s) => ({ ...s, reconGain1: msg.payload }));
          } else if (kind === "recon_low") {
            setState((s) => ({ ...s, reconLow: msg.payload }));
          } else {
            setState((s) => ({ ...s, sample: msg.payload }));
          }
        }
        break;
      case "raster":
        setState((s) => ({ ...s, raster: msg.payload }));
        break;
      case "spike_frame":
        setState((s) => ({ ...s, spikeFrame: msg.payload }));
        break;
      case "run_state":
        setState((s) => ({ ...s, running: msg.payload.running }));
        break;
      case "status":
        if (typeof msg.payload === "object" && msg.payload !== null) {
          setState((s) => ({
            ...s,
            status: msg.payload as StatusPayload,
          }));
        }
        break;
      case "error":
        setState((s) => ({ ...s, error: msg.payload, running: false }));
        break;
    }
  }, []);

  const { connected, send } = useWebSocket({ onMessage: handleMessage });

  const patchConfig = (patch: Partial<EncodeConfig>) =>
    setConfig((c) => ({ ...c, ...patch }));

  const run = () => {
    setState((s) => ({ ...s, error: null }));
    send("configure", config);
    send("run", config);
  };

  const stop = () => {
    setState((s) => ({ ...s, running: false }));
    send("stop", config);
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1>SNN Interpreter</h1>
        <span className={`dot ${connected ? "ok" : "bad"}`} />
        <span>{connected ? "connected" : "disconnected"}</span>
        {state.status && (
          <span className="status">
            {state.status.coding} · {state.status.num_steps} steps
            {state.status.target !== null && state.status.target !== undefined
              ? ` · target ${state.status.target}`
              : ""}
          </span>
        )}
      </header>

      <div className="grid">
        <div className="col-controls">
          <Controls config={config} onChange={patchConfig} />
          <div className="panel actions">
            <button className="apply" onClick={run} disabled={!connected || state.running}>
              {connected ? "▶ Run" : "Connecting…"}
            </button>
            <button className="apply stop" onClick={stop} disabled={!connected || !state.running}>
              ■ Stop
            </button>
          </div>
        </div>

        <div className="col-viz">
          <div className="row">
            <HeatmapCanvas
              data={state.sample}
              palette="binary"
              label="Input sample"
              width={224}
              height={224}
            />
            <HeatmapCanvas
              data={state.spikeFrame}
              palette="plasma"
              label="Spike frame"
              width={224}
              height={224}
            />
            {(state.reconGain1 || state.reconLow) && (
              <div className="panel">
                <div className="panel-title">Reconstruction</div>
                <div className="pair">
                  <HeatmapCanvas
                    data={state.reconGain1}
                    palette="binary"
                    label="Gain=1"
                    width={120}
                    height={120}
                  />
                  <HeatmapCanvas
                    data={state.reconLow}
                    palette="binary"
                    label="Low gain"
                    width={120}
                    height={120}
                  />
                </div>
              </div>
            )}
          </div>

          <RasterCanvas raster={state.raster} label="Spike raster" />
        </div>
      </div>

      {state.error && <div className="error">{state.error}</div>}
    </div>
  );
}
