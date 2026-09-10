interface Props {
  step: number | null;
  numSteps: number;
  playing: boolean;
  onScrub: (step: number) => void;
}

/** Shared time cursor: one scrubber that drives every raster's highlight. */
export function TimeCursor({ step, numSteps, playing, onScrub }: Props) {
  const last = Math.max(0, numSteps - 1);
  const value = Math.min(step ?? 0, last);
  return (
    <div className="panel">
      <div className="panel-title row-title">
        <span>Time cursor</span>
        <span className="cursor-readout">
          {playing ? "live" : "paused"} · t = {value} / {last}
        </span>
      </div>
      <input
        className="cursor-range"
        type="range"
        min={0}
        max={last}
        value={value}
        onChange={(e) => onScrub(Number(e.target.value))}
        aria-label="Time step"
      />
    </div>
  );
}
