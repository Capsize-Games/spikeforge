import type {
  EncodingReportPayload,
  SurrogateCurvePayload,
  TrajectoryMetricsPayload,
  TrajectoryPayload,
} from "./introspectionTypes";
import type { NirGraphPayload, NirValidationPayload } from "./nirTypes";
import type {
  BackendRunPayload,
  DeploymentReportPayload,
  TargetListPayload,
} from "./targetTypes";
import type {
  AnimationStatePayload,
  DownloadState,
  InferencePayload,
  RasterPayload,
  RasterSource,
  ServerMsg,
  StatusPayload,
} from "./types";
import {
  applyViewerMessage,
  EMPTY_FRAMES,
  EMPTY_RASTERS,
} from "./viewerStream";

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
  /** Availability of the opt-in hidden-layer animation stream. */
  animation: AnimationStatePayload | null;
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
  /** Deployment target registry with live availability. */
  targetList: TargetListPayload | null;
  /** Last deployment report; only rendered when its target matches. */
  deploymentReport: DeploymentReportPayload | null;
  /** Last executed backend run for the selected target. */
  backendRun: BackendRunPayload | null;
}

export type ViewerAction =
  | { kind: "message"; msg: ServerMsg }
  | { kind: "autoPredict"; value: boolean }
  | { kind: "modelLoading"; value: boolean }
  | { kind: "clearInference" };

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
    animation: null,
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
    targetList: null,
    deploymentReport: null,
    backendRun: null,
  };
}

/** Fold one viewer action into state (pure). */
export function reduceViewer(
  state: ViewerState,
  action: ViewerAction,
): ViewerState {
  switch (action.kind) {
    case "message":
      return applyViewerMessage(state, action.msg);
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
