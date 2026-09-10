import { HELP } from "../helpText";
import { HelpTip } from "./HelpTip";
import { SampleIndex } from "./SampleIndex";

interface Props {
  step: number | null;
  numSteps: number;
  playing: boolean;
  sampleIndex: number;
  onScrub: (step: number) => void;
  onPlay: () => void;
  onStop: () => void;
  onSelectSample: (index: number) => void;
}

/** Shared time cursor: header controls, one scrubber, and a status line. */
export function TimeCursor({
  step,
  numSteps,
  playing,
  sampleIndex,
  onScrub,
  onPlay,
  onStop,
  onSelectSample,
}: Props) {
  const last = Math.max(0, numSteps - 1);
  const value = Math.min(step ?? 0, last);

  return (
    <div className="panel time-cursor">
      <div className="panel-title row-title">
        <span>
          Time cursor
          <HelpTip text={HELP.time_cursor} />
        </span>
        <span className="panel-actions">
          <HelpTip text={HELP.sample_index} />
          <button
            type="button"
            className="icon-btn"
            onClick={onPlay}
            disabled={playing}
            title="Play spike frames"
            aria-label="Play spike frames"
          >
            ▶
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={onStop}
            disabled={!playing}
            title="Stop"
            aria-label="Stop"
          >
            ■
          </button>
          <SampleIndex value={sampleIndex} onChange={onSelectSample} />
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
      <div className="cursor-readout">
        {playing ? "live" : "paused"} · t = {value} / {last}
      </div>
    </div>
  );
}
