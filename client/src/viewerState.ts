import type {
  EncodingReportPayload,
  SurrogateCurvePayload,
  TrajectoryMetricsPayload,
  TrajectoryPayload,
} from "./introspectionTypes";
import type { NirGraphPayload, NirValidationPayload } from "./nirTypes";
import type {
  DownloadState,
  InferencePayload,
  RasterPayload,
  RasterSource,
  ServerMsg,
  StatusPayload,
} from "./types";

/** Spike frames stream for input/hidden only; output is raster-only. */
export type FrameSource = "input" | "hidden";

export type RasterMap = Record<RasterSource, RasterPayload | null>;
export type FrameMap = Record<FrameSource, number[][] | null>;

export interface ViewerState {
  autoPredict: boolean;
  modelLoading: boolean;
  sample: number[][] | null;
  /** Whole-sample ON/OFF frame for event-modality datasets, else null. */
  eventFrame: number[][] | null;
  spikeFrames: FrameMap;
  spikeStep: number | null;
  reconGain1: number[][] | null;
  reconLow: number[][] | null;
  rasters: RasterMap;
  inference: InferencePayload | null;
  status: StatusPayload | null;
  error: string | null;
  running: boolean;
  framesByStep: Record<number, number[][]>;
  download: DownloadState | null;
  /** Educational-mode U[t]/I[t]/S[t] capture for the active model. */
  trajectory: TrajectoryPayload | null;
  /** Educational-mode aggregate metrics for the active model. */
  metrics: TrajectoryMetricsPayload | null;
  /** Encoding report for the currently configured sample. */
  encodingReport: EncodingReportPayload | null;
  /** Derivative curve of the selected surrogate gradient. */
  surrogateCurve: SurrogateCurvePayload | null;
  nirGraph: NirGraphPayload | null;
  nirValidation: NirValidationPayload | null;
}

export type ViewerAction =
  | { kind: "message"; msg: ServerMsg }
  | { kind: "autoPredict"; value: boolean }
  | { kind: "modelLoading"; value: boolean }
  | { kind: "clearInference" };

const EMPTY_RASTERS: RasterMap = { input: null, hidden: null, output: null };
const EMPTY_FRAMES: FrameMap = { input: null, hidden: null };

export function createInitialViewerState(autoPredict: boolean): ViewerState {
  return {
    autoPredict,
    modelLoading: false,
    sample: null,
    eventFrame: null,
    spikeFrames: EMPTY_FRAMES,
    spikeStep: null,
    reconGain1: null,
    reconLow: null,
    rasters: EMPTY_RASTERS,
    inference: null,
    status: null,
    error: null,
    running: false,
    framesByStep: {},
    download: null,
    trajectory: null,
    metrics: null,
    encodingReport: null,
    surrogateCurve: null,
    nirGraph: null,
    nirValidation: null,
  };
}

/** Clear the streamed sample/frames while keeping run status and prefs. */
function clearStream(state: ViewerState): ViewerState {
  return {
    ...state,
    error: null,
    sample: null,
    eventFrame: null,
    spikeFrames: EMPTY_FRAMES,
    spikeStep: null,
    reconGain1: null,
    reconLow: null,
    rasters: EMPTY_RASTERS,
    inference: null,
    framesByStep: {},
    // A new sample invalidates the captured trajectory and metrics; the user
    // refreshes. The surrogate curve is coding-independent, so it survives.
    trajectory: null,
    metrics: null,
    encodingReport: null,
  };
}

function applyImage(
  state: ViewerState,
  kind: string | undefined,
  payload: number[][],
): ViewerState {
  if (kind === "recon_gain1") return { ...state, reconGain1: payload };
  if (kind === "recon_low") return { ...state, reconLow: payload };
  if (kind === "event_frame") return { ...state, eventFrame: payload };
  return { ...state, sample: payload };
}

function applyRaster(
  state: ViewerState,
  source: RasterSource | undefined,
  payload: RasterPayload,
): ViewerState {
  const key = source ?? "input";
  return { ...state, rasters: { ...state.rasters, [key]: payload } };
}

function applySpikeFrame(
  state: ViewerState,
  source: RasterSource | undefined,
  step: number | undefined,
  payload: number[][],
): ViewerState {
  const frameSource: FrameSource = source === "hidden" ? "hidden" : "input";
  const framesByStep =
    frameSource === "input" && step !== undefined
      ? { ...state.framesByStep, [step]: payload }
      : state.framesByStep;
  return {
    ...state,
    spikeFrames: { ...state.spikeFrames, [frameSource]: payload },
    framesByStep,
    spikeStep:
      frameSource === "input" ? step ?? state.spikeStep : state.spikeStep,
  };
}

function applyStatus(
  state: ViewerState,
  payload: StatusPayload | string,
): ViewerState {
  if (typeof payload !== "object" || payload === null) return state;
  return { ...state, status: payload };
}

function applyMessage(state: ViewerState, msg: ServerMsg): ViewerState {
  switch (msg.type) {
    case "config_ack":
      return clearStream(state);
    case "image":
      return applyImage(state, msg.kind, msg.payload);
    case "raster":
      return applyRaster(state, msg.source, msg.payload);
    case "spike_frame":
      return applySpikeFrame(state, msg.source, msg.step, msg.payload);
    case "inference":
      return state.autoPredict ? { ...state, inference: msg.payload } : state;
    case "run_state":
      return { ...state, running: msg.payload.running };
    case "status":
      return applyStatus(state, msg.payload);
    case "download_state":
      return { ...state, download: msg.payload };
    case "trajectory":
      return { ...state, trajectory: msg.payload };
    case "metrics":
      return { ...state, metrics: msg.payload };
    case "encoding_report":
      return { ...state, encodingReport: msg.payload };
    case "surrogate_curve":
      return { ...state, surrogateCurve: msg.payload };
    case "nir_graph":
      return { ...state, nirGraph: msg.payload };
    case "nir_validation":
      return { ...state, nirValidation: msg.payload };
    case "error":
      return { ...state, error: msg.payload, running: false };
    default:
      return state;
  }
}

/** Fold one viewer action into state (pure). */
export function reduceViewer(
  state: ViewerState,
  action: ViewerAction,
): ViewerState {
  switch (action.kind) {
    case "message":
      return applyMessage(state, action.msg);
    case "autoPredict":
      return action.value
        ? { ...state, autoPredict: true }
        : { ...state, autoPredict: false, inference: null };
    case "modelLoading":
      return { ...state, modelLoading: action.value };
    case "clearInference":
      return { ...state, inference: null };
  }
}
