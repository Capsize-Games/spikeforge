import { HeatmapCanvas } from "./HeatmapCanvas";

interface Props {
  frame: number[][] | null;
  /** Time step the current frame belongs to, when known. */
  step: number | null;
  /** Named reason the animation is unavailable, shown instead of a plot. */
  unavailable: string | null;
}

/** The per-step hidden-layer activation frame streamed during playback. */
export function HiddenFramePanel({ frame, step, unavailable }: Props) {
  const suffix = step !== null ? ` · t = ${step}` : "";
  return (
    <div className="raster-row">
      <div className="raster-row-head">
        <span className="raster-row-label">
          Hidden frame · per step{suffix}
        </span>
        {(!frame || unavailable) && (
          <span className="raster-row-summary">
            {unavailable ?? "run to stream hidden frames"}
          </span>
        )}
      </div>
      {frame && (
        <HeatmapCanvas
          data={frame}
          palette="plasma"
          width={600}
          height={64}
          fluid
        />
      )}
    </div>
  );
}
