import { useState } from "react";

/** Draft stage-name state for adding a per-stage neuron override. */
export function useStageNeurons(add: (stage: string) => void) {
  const [draft, setDraft] = useState("");

  const submit = () => {
    const stage = draft.trim();
    if (stage) add(stage);
    setDraft("");
  };

  return { draft, setDraft, submit };
}
