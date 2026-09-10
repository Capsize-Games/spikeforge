interface Props {
  running: boolean;
  connected: boolean;
  onTrain: () => void;
  onStop: () => void;
}

/** Train & inspect section content: training actions on a single row. */
export function TrainControls({
  running,
  connected,
  onTrain,
  onStop,
}: Props) {
  const busy = !connected || running;

  return (
    <div className="actions" data-tour="train-controls">
      <button className="apply" onClick={onTrain} disabled={busy}>
        {running ? "Training…" : "⚡ Train model"}
      </button>
      <button
        className="apply stop"
        onClick={onStop}
        disabled={!connected || !running}
      >
        ■ Stop training
      </button>
    </div>
  );
}
