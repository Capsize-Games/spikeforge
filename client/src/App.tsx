import { useCallback, useEffect, useRef, useState } from "react";

import { Controls } from "./components/Controls";
import { ModelPanel } from "./components/ModelPanel";
import { ResourceMonitor } from "./components/ResourceMonitor";
import { TopBar } from "./components/TopBar";
import { TrainingPanel } from "./components/TrainingPanel";
import { ViewerPanels } from "./components/ViewerPanels";
import type {
  EncodeConfig,
  InferencePayload,
  RasterPayload,
  RasterSource,
  ServerMsg,
  StatusPayload,
  SystemStatsPayload,
} from "./types";
import { defaultConfig } from "./types";
import { useTraining } from "./useTraining";
import { useWebSocket } from "./useWebSocket";

/** Spike frames are only streamed for input/hidden; output is raster-only. */
type FrameSource = "input" | "hidden";

type RasterMap = Record<RasterSource, RasterPayload | null>;
type FrameMap = Record<FrameSource, number[][] | null>;

interface ViewerState {
  sample: number[][] | null;
  spikeFrames: FrameMap;
  spikeStep: number | null;
  reconGain1: number[][] | null;
  reconLow: number[][] | null;
  rasters: RasterMap;
  inference: InferencePayload | null;
  status: StatusPayload | null;
  error: string | null;
  running: boolean;
}

const emptyRasters: RasterMap = { input: null, hidden: null, output: null };
const emptyFrames: FrameMap = { input: null, hidden: null };

const initial: ViewerState = {
  sample: null,
  spikeFrames: emptyFrames,
  spikeStep: null,
  reconGain1: null,
  reconLow: null,
  rasters: emptyRasters,
  inference: null,
  status: null,
  error: null,
  running: false,
};

export default function App() {
  const [config, setConfig] = useState<EncodeConfig>(defaultConfig);
  const [state, setState] = useState<ViewerState>(initial);
  const [stats, setStats] = useState<SystemStatsPayload | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const training = useTraining();

  // Keep the latest encode config reachable from debounced callbacks.
  const configRef = useRef(config);
  useEffect(() => {
    configRef.current = config;
  }, [config]);

  const handleMessage = useCallback(
    (msg: ServerMsg) => {
      if (
        msg.type === "model_loaded" ||
        msg.type === "model_cleared" ||
        msg.type === "error"
      ) {
        setModelLoading(false);
      }
      if (training.handleMessage(msg)) return;
      switch (msg.type) {
        case "config_ack":
          setState((s) => ({
            ...s,
            error: null,
            sample: null,
            spikeFrames: emptyFrames,
            spikeStep: null,
            reconGain1: null,
            reconLow: null,
            rasters: emptyRasters,
            inference: null,
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
        case "raster": {
          const source: RasterSource = msg.source ?? "input";
          setState((s) => ({
            ...s,
            rasters: { ...s.rasters, [source]: msg.payload },
          }));
          break;
        }
        case "spike_frame": {
          const source: FrameSource =
            msg.source === "hidden" ? "hidden" : "input";
          setState((s) => ({
            ...s,
            spikeFrames: { ...s.spikeFrames, [source]: msg.payload },
            spikeStep:
              source === "input" ? (msg.step ?? s.spikeStep) : s.spikeStep,
          }));
          break;
        }
        case "inference":
          setState((s) => ({ ...s, inference: msg.payload }));
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
        case "system_stats":
          setStats(msg.payload);
          break;
        case "error":
          setState((s) => ({ ...s, error: msg.payload, running: false }));
          break;
      }
    },
    [training],
  );

  const {
    connected,
    send,
    sendTrain,
    sendNamed,
    sendSelectSample,
    sendInfer,
    sendStats,
  } = useWebSocket({ onMessage: handleMessage });

  useEffect(() => {
    if (!connected) return;
    sendNamed("list_models", "");
    // Populate the viewer immediately so every panel is shown at rest.
    send("configure", configRef.current);
  }, [connected, sendNamed, send]);

  useEffect(() => {
    if (!connected) return;
    sendStats();
    const id = setInterval(sendStats, 2000);
    return () => clearInterval(id);
  }, [connected, sendStats]);

  const patchConfig = (patch: Partial<EncodeConfig>) =>
    setConfig((c) => ({ ...c, ...patch }));

  /** Patch the encode config and rebuild the server-side sample. */
  const selectSample = (patch: Partial<EncodeConfig>) => {
    const next = { ...configRef.current, ...patch };
    configRef.current = next;
    setConfig(next);
    sendSelectSample(next);
  };

  /** Score the currently displayed sample with the loaded model. */
  const runInference = () => {
    sendInfer(configRef.current, training.state.config);
  };

  /** Replay the input spike-frame animation from step 0. */
  const playPreview = () => send("run", configRef.current);

  /** Stop the spike-frame animation. */
  const stopPreview = () => send("stop", configRef.current);

  /** Data settings are shared, so training mirrors the single config. */
  const sharedPatch = {
    dataset: config.dataset,
    subset: config.subset,
    batch_size: config.batch_size,
    num_steps: config.num_steps,
    encode: config,
  };

  const train = () => {
    const merged = { ...training.state.config, ...sharedPatch };
    training.patch(sharedPatch);
    training.reset();
    training.setRunning(true);
    sendTrain("train", merged);
  };

  const stopTrain = () => {
    training.setRunning(false);
    sendTrain("stop_train", training.state.config);
  };

  const saveModel = (name: string) => sendNamed("save_model", name);

  const loadModel = (name: string) => {
    setModelLoading(true);
    training.patch({ checkpoint: name });
    const merged = {
      ...training.state.config,
      ...sharedPatch,
      checkpoint: name,
    };
    sendTrain("load_model", merged, name);
  };

  const newModel = () => sendNamed("new_model", "");

  // Unknown until stats arrive: allow GPU so the option isn't disabled.
  const gpuAvailable = stats
    ? stats.device.available.includes("gpu")
    : true;
  const compatibility = training.state.loaded?.compatibility;
  const hasModel =
    training.state.loaded !== null || training.state.last !== null;
  const canInfer = connected && hasModel && (compatibility?.dataset_match ?? true);

  return (
    <div className="app">
      <header className="app-header">
        <TopBar />
      </header>

      <main className="app-main">
        <div className="grid">
        <div className="col-controls">
          <ModelPanel
            models={training.state.models}
            current={training.state.loaded?.name ?? null}
            connected={connected}
            busy={training.state.running}
            loading={modelLoading}
            onNew={newModel}
            onLoad={loadModel}
            onSave={saveModel}
          />
          <Controls
            config={config}
            model={training.state.config}
            datasets={training.state.datasets}
            gpuAvailable={gpuAvailable}
            connected={connected}
            trainRunning={training.state.running}
            canInfer={canInfer}
            onChange={patchConfig}
            onModelChange={training.patch}
            onSelectSample={selectSample}
            onTrain={train}
            onStopTrain={stopTrain}
            onInfer={runInference}
          />
        </div>

        <ViewerPanels
          sample={state.sample}
          spikeFrame={state.spikeFrames.input}
          reconGain1={state.reconGain1}
          reconLow={state.reconLow}
          rasters={state.rasters}
          inference={state.inference}
          spikeStep={state.spikeStep}
          playing={state.running}
          onPlay={playPreview}
          onStop={stopPreview}
        />

          <TrainingPanel
            loss={training.state.loss}
            trainAccuracy={training.state.trainAccuracy}
            testAccuracy={training.state.testAccuracy}
            last={training.state.last}
            device={training.state.device}
            inference={state.inference}
            prediction={training.state.prediction}
            loaded={training.state.loaded}
          />
        </div>

        {state.error && <div className="error">{state.error}</div>}
      </main>

      <footer className="app-footer">
        <ResourceMonitor
          stats={stats}
          requested={training.state.config.device}
        />
        <span className={`conn ${connected ? "ok" : "bad"}`}>
          <span className={`dot ${connected ? "ok" : "bad"}`} />
          {connected ? "connected" : "disconnected"}
        </span>
      </footer>
    </div>
  );
}
