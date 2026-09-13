interface Props {
  running: boolean;
  connected: boolean;
  readOnly?: boolean;
  onTrain: () => void;
  onStop: () => void;
}

/** Train & inspect section content: training actions on a single row. */
export function TrainControls({
  running,
  connected,
  readOnly = false,
  onTrain,
  onStop,
}: Props) {
  const busy = !connected || running || readOnly;

  return (
    <div className="actions" data-tour="train-controls">
      <button className="apply" onClick={onTrain} disabled={busy}>
        {readOnly
          ? "Training disabled"
          : running
            ? "Training…"
            : "⚡ Train model"}
      </button>
      {readOnly && (
        <p className="arch-note">Training is disabled on the public demo.</p>
      )}
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
