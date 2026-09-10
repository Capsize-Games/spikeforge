import type { ModelLoadedPayload } from "../types";
import { TRAIN_HELP } from "../helpText";
import { HelpTip } from "./HelpTip";

/** Compact one-row summary of the currently loaded checkpoint. */
export function LoadedModelPanel({ loaded }: { loaded: ModelLoadedPayload }) {
  return (
    <div className="panel loaded-model">
      <div className="loaded-model-row">
        <span className="loaded-model-name">{loaded.name}</span>
        <span className="loaded-model-sep">·</span>
        <span>{loaded.dataset}</span>
        <span className="loaded-model-sep">·</span>
        <span>acc {loaded.accuracy.toFixed(1)}%</span>
        <span className="loaded-model-sep">·</span>
        <span>input {loaded.input_mode ?? "raw"}</span>
        <span className="loaded-model-sep">·</span>
        <span>hidden {loaded.hidden ?? "—"}</span>
        <span className="loaded-model-sep">·</span>
        <span>device {loaded.device ?? "—"}</span>
        <HelpTip text={TRAIN_HELP.input_mode} />
      </div>
    </div>
  );
}
