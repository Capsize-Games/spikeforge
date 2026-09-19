import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";

import { STORAGE_KEYS } from "./storage";

export type Theme = "dark" | "light";

/**
 * Canvas drawing colors, which cannot be driven by CSS variables. This is the
 * plot palette: icy cyan primary, violet-blue secondary trace, and a
 * desaturated slate for gridlines and reference marks.
 */
export interface CanvasColors {
  bg: string;
  grid: string;
  text: string;
  accent: string;
  /** Secondary trace colour, paired with `accent` in two-series plots. */
  trace: string;
  accentSoft: string;
  dot: string;
  /** Translucent band drawn behind highlighted rows. */
  highlight: string;
}

export const CANVAS_COLORS: Record<Theme, CanvasColors> = {
  dark: {
    bg: "#05070c",
    grid: "#1b2231",
    text: "#93a3bd",
    accent: "#70cfff",
    trace: "#8b95ff",
    accentSoft: "rgba(112, 207, 255, 0.16)",
    dot: "#e3b341",
    highlight: "rgba(63, 185, 80, 0.14)",
  },
  light: {
    bg: "#eef1f5",
    grid: "#e7ebf0",
    text: "#57606a",
    accent: "#0969da",
    trace: "#6f5bd6",
    accentSoft: "rgba(9, 105, 218, 0.12)",
    dot: "#9a6700",
    highlight: "rgba(26, 127, 55, 0.14)",
  },
};

function readTheme(): Theme {
  try {
    const t = localStorage.getItem(STORAGE_KEYS.theme);
    if (t === "light" || t === "dark") return t;
  } catch {
    /* ignore */
  }
  return "dark";
}

/** Stamp the stored theme onto <html> before React mounts (avoids a flash). */
export function applyStoredTheme(): void {
  document.documentElement.dataset.theme = readTheme();
}

interface ThemeContextValue {
  theme: Theme;
  colors: CanvasColors;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "dark",
  colors: CANVAS_COLORS.dark,
  toggle: () => undefined,
});

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(readTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(STORAGE_KEYS.theme, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const toggle = useCallback(
    () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    [],
  );

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, colors: CANVAS_COLORS[theme], toggle }),
    [theme, toggle],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
