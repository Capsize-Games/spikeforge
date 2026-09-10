import { useEffect, useState } from "react";

/** Track the selected deployment target, keeping it valid as names load. */
export function useTargetSelection(names: string[]): {
  target: string;
  setTarget: (value: string) => void;
} {
  const [target, setTarget] = useState("reference");

  useEffect(() => {
    if (names.length > 0 && !names.includes(target)) {
      setTarget(names.includes("reference") ? "reference" : names[0]);
    }
  }, [names, target]);

  return { target, setTarget };
}
