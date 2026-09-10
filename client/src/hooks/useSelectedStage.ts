import { useEffect, useState } from "react";

/** Track the selected trajectory stage, keeping it valid as stages change. */
export function useSelectedStage(stages: string[]): {
  stage: string;
  setStage: (value: string) => void;
} {
  const [stage, setStage] = useState(stages[0] ?? "");

  useEffect(() => {
    if (stages.length > 0 && !stages.includes(stage)) {
      setStage(stages[0]);
    }
  }, [stages, stage]);

  return { stage, setStage };
}
