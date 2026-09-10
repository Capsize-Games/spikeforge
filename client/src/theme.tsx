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

/** Canvas drawing colors, which cannot be driven by CSS variables. */
export interface CanvasColors {
  bg: string;
  grid: string;
  text: string;
  accent: string;
  accentSoft: string;
  dot: string;
}

export const CANVAS_COLORS: Record<Theme, CanvasColors> = {
  dark: {
    bg: "#0d1117",
    grid: "#30363d",
    text: "#8b949e",
    accent: "#79c0ff",
    accentSoft: "rgba(88, 166, 255, 0.16)",
    dot: "#e3b341",
  },
  light: {
    bg: "#ffffff",
    grid: "#d0d7de",
    text: "#59636e",
    accent: "#0969da",
    accentSoft: "rgba(9, 105, 218, 0.12)",
    dot: "#9a6700",
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

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
