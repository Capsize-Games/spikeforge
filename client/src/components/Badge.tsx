import type { ReactNode } from "react";

/** Semantic tone; `neutral` is the uncoloured default. */
export type BadgeTone = "neutral" | "ok" | "warn" | "bad";

interface Props {
  children: ReactNode;
  tone?: BadgeTone;
  title?: string;
}

/** A short status marker: a verdict, a mode, a cached/downloaded state. */
export function Badge({ children, tone = "neutral", title }: Props) {
  return (
    <span
      className={tone === "neutral" ? "badge" : `badge ${tone}`}
      title={title}
    >
      {children}
    </span>
  );
}
