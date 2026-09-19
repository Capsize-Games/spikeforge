import { useState } from "react";
import type { ReactNode } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { useI18n } from "../i18n/I18nProvider";
import { IconButton } from "./IconButton";

/** A pane's width policy: fixed for a browser or an inspector, free
 * otherwise. */
type PaneVariant = "assets" | "editor" | "inspector";

/**
 * Viewport width below which a pane starts collapsed, by role.
 *
 * The order is deliberate and follows each pane's weight: the asset browser
 * gives way first, the inspector second, and the workspace only stacks much
 * later (see dock.css). The editor is never in this table — it is where the
 * work happens, so it keeps whatever width is left.
 */
const COLLAPSE_BELOW: Partial<Record<PaneVariant, number>> = {
  assets: 1200,
  inspector: 1040,
};

/**
 * Whether a pane is open at first paint.
 *
 * Read once, from the viewport, because a pane the user has expanded by hand
 * should stay expanded: re-deriving it on every resize would fight the
 * toggle. A pane with no entry in the table is always open.
 */
function initiallyOpen(variant: PaneVariant): boolean {
  const below = COLLAPSE_BELOW[variant];
  return below === undefined || window.innerWidth > below;
}

interface Props {
  variant: PaneVariant;
  /** Pane heading, rendered in the head row. */
  heading: string;
  /** Small right-aligned fact for the head row, e.g. the current dataset. */
  meta?: string;
  children: ReactNode;
}

/**
 * One docked pane: a full-height child of a workspace, separated from its
 * neighbour by a single hairline. It has no margin, no radius, and no card on
 * a background — the surface step and the hairline are the whole hierarchy.
 *
 * The head is a fixed row and the body scrolls beneath it, so a long form
 * keeps its pane heading and a short one keeps its empty space inside the
 * pane instead of leaving a gap at the bottom of the window.
 *
 * An asset browser or an inspector collapses to a labelled strip on a narrow
 * desktop and expands again with its toggle. The body is hidden rather than
 * unmounted, so nothing inside a collapsed pane is torn down or loses state.
 */
export function DockPane({ variant, heading, meta, children }: Props) {
  const { t } = useI18n();
  const collapsible = COLLAPSE_BELOW[variant] !== undefined;
  const [open, setOpen] = useState(() => initiallyOpen(variant));

  return (
    <section
      className={`dock-pane dock-pane--${variant}${open ? "" : " collapsed"}`}
    >
      <div className="pane-head">
        <span className="pane-title">{heading}</span>
        {meta !== undefined && <span className="pane-head-meta">{meta}</span>}
        {collapsible && (
          <IconButton
            className="pane-toggle"
            icon={open ? ChevronLeft : ChevronRight}
            label={open ? t("pane.collapse") : t("pane.expand")}
            expanded={open}
            onClick={() => setOpen((isOpen) => !isOpen)}
          />
        )}
      </div>
      <div className="pane-body" hidden={!open}>
        {children}
      </div>
    </section>
  );
}
