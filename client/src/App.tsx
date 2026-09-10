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
  TrainConfig,
} from "./types";
import {
  readBool,
  readJson,
  STORAGE_KEYS,
  writeBool,
  writeJson,
} from "./storage";
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
  const [config, setConfig] = useState<EncodeConfig>(() =>
    readJson(STORAGE_KEYS.encode, defaultConfig),
  );
  const [state, setState] = useState<ViewerState>(initial);
  const [stats, setStats] = useState<SystemStatsPayload | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [scrubStep, setScrubStep] = useState<number | null>(null);
  const [framesByStep, setFramesByStep] = useState<
    Record<number, number[][]>
  >({});
  const training = useTraining();
  const [autoPredict, setAutoPredict] = useState<boolean>(() =>
    readBool(STORAGE_KEYS.autoPredict, false),
  );

  // Keep the latest encode config reachable from debounced callbacks.
  const configRef = useRef(config);
  useEffect(() => {
    configRef.current = config;
  }, [config]);

  // Persist encode settings so a page reload restores them.
  useEffect(() => {
    writeJson(STORAGE_KEYS.encode, config);
  }, [config]);

  // Event handlers read auto-predict synchronously, so mirror it into a ref.
  const autoPredictRef = useRef(autoPredict);
  useEffect(() => {
    autoPredictRef.current = autoPredict;
    writeBool(STORAGE_KEYS.autoPredict, autoPredict);
  }, [autoPredict]);

  const handleMessage = useCallback(
    (msg: ServerMsg) => {
      if (
        msg.type === "model_loaded" ||
        msg.type === "model_cleared" ||
        msg.type === "error"
      ) {
        setModelLoading(false);
      }
      // Refresh the network-activity rasters as soon as a model exists.
      const trained =
        msg.type === "train_state" &&
        msg.payload.running === false &&
        msg.payload.reason === "finished";
      // Always refresh the layer-activity rasters when a model appears; only
      // the prediction display itself is gated by the auto-predict toggle.
      if (msg.type === "model_loaded" || trained) {
        window.setTimeout(
          () => sendInfer(configRef.current, training.state.config),
          0,
        );
      }
      if (training.handleMessage(msg)) return;
      switch (msg.type) {
        case "config_ack":
          setScrubStep(null);
          setFramesByStep({});
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
          const step = msg.step;
          setState((s) => ({
            ...s,
            spikeFrames: { ...s.spikeFrames, [source]: msg.payload },
            spikeStep: source === "input" ? (step ?? s.spikeStep) : s.spikeStep,
          }));
          if (source === "input" && step !== undefined) {
            setFramesByStep((m) => ({ ...m, [step]: msg.payload }));
          }
          break;
        }
        case "inference":
          // While auto-predict is off we ignore predictions; the server still
          // streams the layer-activity rasters, which keep updating.
          if (autoPredictRef.current) {
            setState((s) => ({ ...s, inference: msg.payload }));
          }
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

  /** Replay the input spike-frame animation from step 0 (back to live). */
  const playPreview = () => {
    setScrubStep(null);
    send("run", configRef.current);
  };

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

  /** Apply a checkpoint's saved architecture/encoding, then load it. */
  const loadModel = (name: string) => {
    setModelLoading(true);
    const saved = training.state.models.find((m) => m.name === name);
    const meta = (saved?.meta ?? {}) as {
      encode?: Partial<EncodeConfig>;
      dataset?: string;
      hidden?: number;
      beta?: number;
    };
    // Restore the exact encoding the checkpoint was trained with so the
    // server's compatibility check passes, then lock the controls.
    const nextEncode: EncodeConfig = meta.encode
      ? {
          ...defaultConfig,
          ...meta.encode,
          sample_index: configRef.current.sample_index,
        }
      : { ...configRef.current };
    configRef.current = nextEncode;
    setConfig(nextEncode);

    const patch: Partial<TrainConfig> = {
      checkpoint: name,
      encode: nextEncode,
    };
    if (meta.dataset) patch.dataset = meta.dataset;
    if (typeof meta.hidden === "number") patch.hidden = meta.hidden;
    if (typeof meta.beta === "number") patch.beta = meta.beta;
    training.patch(patch);

    const merged: TrainConfig = {
      ...training.state.config,
      ...sharedPatch,
      ...patch,
      checkpoint: name,
    };
    sendTrain("load_model", merged, name);
  };

  const newModel = () => sendNamed("new_model", "");

  // Unknown until stats arrive: allow GPU so the option isn't disabled.
  const gpuAvailable = stats
    ? stats.device.available.includes("gpu")
    : true;

  // Shared time cursor: scrubbed step wins, else follow the live stream.
  const numSteps =
    state.rasters.input?.num_steps ??
    state.status?.num_steps ??
    config.num_steps;
  const timeStep = scrubStep ?? (state.running ? state.spikeStep : null);
  const displayFrame =
    (scrubStep !== null ? framesByStep[scrubStep] : undefined) ??
    state.spikeFrames.input;
  const compatibility = training.state.loaded?.compatibility;
  const hasModel =
    training.state.loaded !== null || training.state.last !== null;
  const canAutoPredict =
    connected && hasModel && (compatibility?.dataset_match ?? true);
  const locked = training.state.loaded !== null;

  /** Toggle auto-prediction: run now when enabled, clear the panel when off. */
  const toggleAutoPredict = () => {
    const next = !autoPredict;
    setAutoPredict(next);
    if (next) {
      if (canAutoPredict) runInference();
    } else {
      setState((s) => ({ ...s, inference: null }));
    }
  };

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
            locked={locked}
            onChange={patchConfig}
            onModelChange={training.patch}
            onSelectSample={selectSample}
            onTrain={train}
            onStopTrain={stopTrain}
          />
        </div>

        <ViewerPanels
          sample={state.sample}
          spikeFrame={displayFrame}
          reconGain1={state.reconGain1}
          rasters={state.rasters}
          inference={state.inference}
          loaded={training.state.loaded}
          coding={config.coding}
          sampleIndex={config.sample_index}
          timeStep={timeStep}
          numSteps={numSteps}
          playing={state.running}
          onPlay={playPreview}
          onStop={stopPreview}
          onScrub={(step) => setScrubStep(step)}
          onSelectSample={(index) => selectSample({ sample_index: index })}
        />

          <TrainingPanel
            loss={training.state.loss}
            trainAccuracy={training.state.trainAccuracy}
            testAccuracy={training.state.testAccuracy}
            last={training.state.last}
            device={training.state.device}
            inference={state.inference}
            timeStep={timeStep}
            loaded={training.state.loaded}
            autoPredict={autoPredict}
            canAutoPredict={canAutoPredict}
            onToggleAutoPredict={toggleAutoPredict}
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
