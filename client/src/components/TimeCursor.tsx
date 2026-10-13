import { Play, Square } from "lucide-react";

import { HELP } from "../helpText";
import { HelpTip } from "./HelpTip";
import { IconButton } from "./IconButton";
import { Metric } from "./Metric";
import { SampleIndex } from "./SampleIndex";
import { Toolbar } from "./Toolbar";

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

/**
 * The viewer's transport strip: play/pause, the shared timestep scrubber, the
 * current step as a tabular readout, and the sample stepper. Everything it
 * moves (every raster, the neuron trace, the prediction bars) follows the one
 * cursor.
 */
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
    <Toolbar className="time-cursor">
      <span className="transport-title">
        Time cursor
        <HelpTip text={HELP.time_cursor} />
      </span>

      <IconButton
        icon={Play}
        label="Play spike frames"
        onClick={onPlay}
        disabled={playing}
      />
      <IconButton
        icon={Square}
        label="Stop"
        onClick={onStop}
        disabled={!playing}
      />

      <input
        className="cursor-range"
        type="range"
        min={0}
        max={last}
        value={value}
        onChange={(e) => onScrub(Number(e.target.value))}
        aria-label="Time step"
      />

      <Metric small label="t" value={`${value} / ${last}`} />
      <span className="cursor-readout">{playing ? "live" : "paused"}</span>

      <HelpTip text={HELP.sample_index} />
      <SampleIndex value={sampleIndex} onChange={onSelectSample} />
    </Toolbar>
  );
}
