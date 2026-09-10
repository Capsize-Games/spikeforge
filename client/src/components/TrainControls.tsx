interface Props {
  running: boolean;
  connected: boolean;
  canInfer: boolean;
  onTrain: () => void;
  onStop: () => void;
  onInfer: () => void;
}

/** Section-4 content: training and prediction actions. */
export function TrainControls({
  running,
  connected,
  canInfer,
  onTrain,
  onStop,
  onInfer,
}: Props) {
  const busy = !connected || running;

  return (
    <>
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
      <button
        className="apply ghost"
        onClick={onInfer}
        disabled={busy || !canInfer}
      >
        🔎 Predict displayed sample
      </button>
    </>
  );
}
