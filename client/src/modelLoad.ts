/** Restore a saved checkpoint's architecture and encoding before loading.

Kept out of `useModelActions` so that hook stays within its 120-line limit.
*/

import { defaultConfig } from "./types";
import type { EncodeConfig, SavedModel, TrainConfig } from "./types";

/** Checkpoint meta fields that seed the encoding controls on load. */
interface ModelMeta {
  encode?: Partial<EncodeConfig>;
  dataset?: string;
  hidden?: number;
  beta?: number;
}

export interface LoadCheckpointDeps {
  /** The picked checkpoint, absent if it vanished from the model list. */
  saved: SavedModel | undefined;
  /** Current encode config, used when the checkpoint saved none. */
  current: EncodeConfig;
  /** Current training config, base for the merged load request. */
  config: TrainConfig;
  /** Shared dataset/encode fields that always mirror the signal config. */
  sharedPatch: Partial<TrainConfig>;
  replaceConfig: (config: EncodeConfig) => void;
  applyPatch: (patch: Partial<TrainConfig>) => void;
}

/** Apply a checkpoint's saved settings and return the load request config. */
export function loadCheckpoint(
  name: string,
  deps: LoadCheckpointDeps,
): TrainConfig {
  const { saved, current, config, sharedPatch } = deps;
  const { replaceConfig, applyPatch } = deps;
  const meta = (saved?.meta ?? {}) as ModelMeta;
  // Restore the exact encoding the checkpoint was trained with so the
  // server's compatibility check passes, then lock the controls.
  const encode: EncodeConfig = meta.encode
    ? {
        ...defaultConfig,
        ...meta.encode,
        sample_index: current.sample_index,
      }
    : { ...current };
  replaceConfig(encode);

  const patch: Partial<TrainConfig> = { checkpoint: name, encode };
  if (meta.dataset) patch.dataset = meta.dataset;
  if (typeof meta.hidden === "number") patch.hidden = meta.hidden;
  if (typeof meta.beta === "number") patch.beta = meta.beta;
  applyPatch(patch);

  return { ...config, ...sharedPatch, ...patch, checkpoint: name };
}
