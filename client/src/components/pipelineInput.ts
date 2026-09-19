/**
 * The run-input template a pipeline starts from.
 *
 * A pipeline feeds its source nodes with `{frames, encoded}`, the same body
 * `spikeforge-serve` reads from stdin. The two modes are not interchangeable:
 *
 * - `encoded: true` steps each frame straight into the first layer. Nothing
 *   reshapes it, so the frame has to already match that layer's expected
 *   input. A flat vector works for a fully connected topology and fails for a
 *   convolutional one, which needs `[C, H, W]`.
 * - `encoded: false` treats each frame as a **raw sample**. The server encodes
 *   it with the checkpoint's own frozen encode spec and lays the result out
 *   for the topology's input stage, so one body works for every topology.
 *
 * The template therefore uses the raw form. A raw sample must be at least 3-D
 * (`[C, H, W]`) or the server reads a 2-D payload as a *batch of rows* rather
 * than one image -- which is what made an earlier hand-written `[H][W]`
 * template fail with a matrix-shape error.
 *
 * One raw frame is a complete run: the encoder expands it into the whole spike
 * train and the session steps every timestep, so the template does not need to
 * repeat itself once per step.
 */

import type { SavedModel } from "../types";

/** Input geometry assumed when a checkpoint recorded none (MNIST-shaped). */
const DEFAULT_GEOMETRY: readonly [number, number] = [28, 28];

/** Metadata a checkpoint may record about the input it expects. */
interface GeometrySource {
  input_size?: unknown;
}

/** Return `value` as a positive integer, or `fallback` when it is not one. */
function positiveInt(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 1) {
    return fallback;
  }
  return Math.floor(value);
}

/** Read `input_size` off a metadata block that may be absent or malformed. */
function geometryFrom(block: unknown): readonly [number, number] | null {
  if (typeof block !== "object" || block === null) return null;
  const value = (block as GeometrySource).input_size;
  if (!Array.isArray(value) || value.length !== 2) return null;
  const [height, width] = value;
  if (typeof height !== "number" || typeof width !== "number") return null;
  return [
    positiveInt(height, DEFAULT_GEOMETRY[0]),
    positiveInt(width, DEFAULT_GEOMETRY[1]),
  ];
}

/**
 * The `[height, width]` a checkpoint was trained on.
 *
 * Checked in the order the engine itself resolves them: the frozen encode
 * spec a bundle carries, then the topology's own parameters, then the
 * top-level field, then the default. Reading only the last of those -- which
 * is null for every dataset that uses the default geometry -- silently
 * produced an MNIST-shaped template for checkpoints that were not.
 */
export function inputGeometry(
  model: SavedModel | undefined,
): readonly [number, number] {
  const meta = model?.meta ?? {};
  return (
    geometryFrom(meta.encode_spec) ??
    geometryFrom(meta.topology_params) ??
    geometryFrom(meta) ??
    DEFAULT_GEOMETRY
  );
}

/**
 * A comparable description of what a checkpoint accepts.
 *
 * Two source nodes fed by one run body must agree on this, or the body cannot
 * satisfy both.
 */
export function inputContract(model: SavedModel | undefined): string {
  const [height, width] = inputGeometry(model);
  return `${height}x${width}`;
}

/** Build a runnable `{frames, encoded}` body for `model`. */
export function inputTemplate(model?: SavedModel): string {
  const [height, width] = inputGeometry(model);
  // One channel, matching the grayscale samples every bundled dataset ships.
  const sample = [
    Array.from({ length: height }, () => new Array(width).fill(0)),
  ];
  return JSON.stringify({ frames: [sample], encoded: false });
}
