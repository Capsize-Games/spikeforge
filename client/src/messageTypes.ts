import type { EncodeConfig } from "./configTypes";
import type { EnergyServerMsg } from "./energyTypes";
import type { HubServerMsg } from "./hubTypes";
import type { IntrospectionServerMsg } from "./introspectionTypes";
import type { NirGraphPayload, NirValidationPayload } from "./nirTypes";
import type {
  AnimationStatePayload,
  DownloadState,
  InferencePayload,
  PredictionPayload,
  RasterPayload,
  RasterSource,
  RunStatePayload,
  StatusPayload,
  SystemStatsPayload,
  TrainMetrics,
  TrainStatePayload,
} from "./payloadTypes";
import type { PipelineServerMsg } from "./pipelineTypes";
import type {
  Modality,
  ModelListPayload,
  ModelLoadedPayload,
  ModelRegistryServerMsg,
} from "./protocolTypes";
import type { TargetServerMsg } from "./targetTypes";

export type ServerMsg =
  | { type: "config_ack"; payload: EncodeConfig & { modality?: Modality } }
  | { type: "status"; payload: StatusPayload | string }
  | { type: "image"; payload: number[][]; kind?: string }
  | { type: "raster"; payload: RasterPayload; source?: RasterSource }
  | { type: "spike_frame"; payload: number[][]; step?: number;
      source?: RasterSource }
  | { type: "run_state"; payload: RunStatePayload }
  | { type: "train_metrics"; payload: TrainMetrics }
  | { type: "train_state"; payload: TrainStatePayload }
  | { type: "prediction"; payload: PredictionPayload }
  | { type: "inference"; payload: InferencePayload }
  | { type: "animation_state"; payload: AnimationStatePayload }
  | { type: "nir_graph"; payload: NirGraphPayload }
  | { type: "nir_validation"; payload: NirValidationPayload }
  | { type: "model_saved"; payload: { name: string; path: string } }
  | { type: "model_list"; payload: ModelListPayload }
  | { type: "model_loaded"; payload: ModelLoadedPayload }
  | { type: "model_cleared"; payload?: null }
  | { type: "system_stats"; payload: SystemStatsPayload }
  | { type: "download_state"; payload: DownloadState }
  | IntrospectionServerMsg
  | TargetServerMsg
  | ModelRegistryServerMsg
  | HubServerMsg
  | EnergyServerMsg
  | PipelineServerMsg
  | { type: "error"; payload: string };
