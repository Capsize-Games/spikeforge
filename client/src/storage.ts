/** localStorage helpers that never throw (private mode / quota safe). */

export const STORAGE_KEYS = {
  encode: "snn.encodeConfig",
  train: "snn.trainConfig",
  theme: "snn.theme",
  autoPredict: "snn.autoPredict",
} as const;

/** Read a persisted JSON object and shallow-merge it over the fallback. */
export function readJson<T extends object>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<T>;
    return { ...fallback, ...parsed };
  } catch {
    return fallback;
  }
}

/** Persist a value as JSON, ignoring quota or privacy-mode errors. */
export function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

export function readBool(key: string, fallback: boolean): boolean {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return raw === "true";
  } catch {
    return fallback;
  }
}

export function writeBool(key: string, value: boolean): void {
  try {
    localStorage.setItem(key, value ? "true" : "false");
  } catch {
    /* ignore */
  }
}
