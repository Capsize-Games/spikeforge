import { useEffect, useRef, useState } from "react";

/**
 * Track the selected surrogate gradient and auto-sample its derivative.
 *
 * The request callback is held in a ref so an unstable inlining never
 * re-fires the sampling effect; only an actual selection change does.
 */
export function useSurrogateCurve(
  current: string,
  request: (name: string) => void,
): {
  value: string;
  setValue: (value: string) => void;
  reload: () => void;
} {
  const [value, setValue] = useState(current);
  const requestRef = useRef(request);

  useEffect(() => {
    requestRef.current = request;
  }, [request]);

  // Follow the configured surrogate when the training config changes.
  useEffect(() => {
    setValue(current);
  }, [current]);

  // Sample whenever a named surrogate is selected; "" is the default.
  useEffect(() => {
    if (value) requestRef.current(value);
  }, [value]);

  const reload = () => {
    if (value) requestRef.current(value);
  };

  return { value, setValue, reload };
}
