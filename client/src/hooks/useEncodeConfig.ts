import { useCallback, useEffect, useRef, useState } from "react";

import { readJson, STORAGE_KEYS, writeJson } from "../storage";
import { defaultConfig } from "../types";
import type { EncodeConfig } from "../types";

/** Own the encode config, a ref mirror, and its persistence. */
export function useEncodeConfig() {
  const [config, setConfig] = useState<EncodeConfig>(() =>
    readJson(STORAGE_KEYS.encode, defaultConfig),
  );
  const configRef = useRef(config);

  useEffect(() => {
    configRef.current = config;
    writeJson(STORAGE_KEYS.encode, config);
  }, [config]);

  const patchConfig = useCallback((patch: Partial<EncodeConfig>) => {
    setConfig((current) => ({ ...current, ...patch }));
  }, []);

  /** Replace the whole config (used when a checkpoint is loaded). */
  const replaceConfig = useCallback((next: EncodeConfig) => {
    configRef.current = next;
    setConfig(next);
  }, []);

  return { config, configRef, patchConfig, replaceConfig };
}
