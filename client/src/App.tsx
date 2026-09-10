import { useCallback, useEffect, useState } from "react";

import { Controls } from "./components/Controls";
import { TopBar } from "./components/TopBar";
import { TrainingPanel } from "./components/TrainingPanel";
import { ViewerPanels } from "./components/ViewerPanels";
import type {
  EncodeConfig,
  RasterPayload,
  ServerMsg,
  StatusPayload,
} from "./types";
import { defaultConfig } from "./types";
import { useTraining } from "./useTraining";
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
  const training = useTraining();

  const handleMessage = useCallback(
    (msg: ServerMsg) => {
      if (training.handleMessage(msg)) return;
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
        case "image": {
          const kind = (msg as { kind?: string }).kind;
          if (kind === "recon_gain1") {
            setState((s) => ({ ...s, reconGain1: msg.payload }));
          } else if (kind === "recon_low") {
            setState((s) => ({ ...s, reconLow: msg.payload }));
          } else {
            setState((s) => ({ ...s, sample: msg.payload }));
          }
          break;
        }
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
    },
    [training],
  );

  const { connected, send, sendTrain, sendNamed } = useWebSocket({
    onMessage: handleMessage,
  });

  useEffect(() => {
    if (connected) sendNamed("list_models", "");
  }, [connected, sendNamed]);

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

  const train = () => {
    training.reset();
    training.setRunning(true);
    sendTrain("train", training.state.config);
  };

  const stopTrain = () => {
    training.setRunning(false);
    sendTrain("stop_train", training.state.config);
  };

  const predict = () => sendTrain("predict", training.state.config);

  const saveModel = (name: string) =>
    sendNamed("save_model", name);

  const loadModel = (name: string) => {
    training.patch({ checkpoint: name });
    sendTrain("load_model", { ...training.state.config, checkpoint: name }, name);
  };

  const deleteModel = (name: string) => sendNamed("delete_model", name);

  return (
    <div className="app">
      <TopBar connected={connected} status={state.status} />

      <div className="grid">
        <div className="col-controls">
          <Controls config={config} onChange={patchConfig} />
          <div className="panel actions">
            <button
              className="apply"
              onClick={run}
              disabled={!connected || state.running}
            >
              {connected ? "▶ Run" : "Connecting…"}
            </button>
            <button
              className="apply stop"
              onClick={stop}
              disabled={!connected || !state.running}
            >
              ■ Stop
            </button>
          </div>
        </div>

        <ViewerPanels
          sample={state.sample}
          spikeFrame={state.spikeFrame}
          reconGain1={state.reconGain1}
          reconLow={state.reconLow}
          raster={state.raster}
        />

        <TrainingPanel
          config={training.state.config}
          datasets={training.state.datasets}
          models={training.state.models}
          onChange={training.patch}
          onTrain={train}
          onStop={stopTrain}
          onPredict={predict}
          onSave={saveModel}
          onLoad={loadModel}
          onDelete={deleteModel}
          running={training.state.running}
          connected={connected}
          loss={training.state.loss}
          trainAccuracy={training.state.trainAccuracy}
          testAccuracy={training.state.testAccuracy}
          last={training.state.last}
          prediction={training.state.prediction}
          loaded={training.state.loaded}
        />
      </div>

      {state.error && <div className="error">{state.error}</div>}
    </div>
  );
}
