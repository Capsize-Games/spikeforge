import type {
  RasterPayload,
  RasterSource,
  ServerMsg,
  StatusPayload,
} from "./types";
import type { FrameMap, FrameSource, RasterMap } from "./viewerState";
import type { ViewerState } from "./viewerState";

export const EMPTY_RASTERS: RasterMap = {
  input: null,
  hidden: null,
  output: null,
};
export const EMPTY_FRAMES: FrameMap = { input: null, hidden: null };

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
    animation: null,
    framesByStep: {},
    // A new sample invalidates the captured trajectory and metrics; the user
    // refreshes. The surrogate curve is coding-independent, so it survives.
    trajectory: null,
    metrics: null,
    encodingReport: null,
    // The report classifies the current topology, so a reconfigure stales it;
    // the registry itself is topology-independent and survives.
    deploymentReport: null,
    backendRun: null,
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

/**
 * Extract a display string from an error payload. The wire schema allows
 * either a plain reason string or a structured object (for example a
 * `protocol_version_mismatch`, which carries `code`/`client`/`server`
 * alongside `message`) -- rendering the object directly crashes React.
 */
function errorMessage(payload: unknown): string {
  if (typeof payload === "string") return payload;
  if (
    payload !== null &&
    typeof payload === "object" &&
    typeof (payload as { message?: unknown }).message === "string"
  ) {
    return (payload as { message: string }).message;
  }
  return "Unknown error";
}

/** Fold one server message into viewer state (pure). */
export function applyViewerMessage(
  state: ViewerState,
  msg: ServerMsg,
): ViewerState {
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
    case "animation_state":
      return { ...state, animation: msg.payload };
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
    case "target_list":
      return { ...state, targetList: msg.payload };
    case "deployment_report":
      return { ...state, deploymentReport: msg.payload };
    case "backend_run":
      return { ...state, backendRun: msg.payload };
    case "error":
      return { ...state, error: errorMessage(msg.payload), running: false };
    default:
      return state;
  }
}
