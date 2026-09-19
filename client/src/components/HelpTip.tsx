import { useRef, useState } from "react";
import { createPortal } from "react-dom";

interface Props {
  text: string;
}

/** Widest the popup is allowed to be before it is clamped to the viewport. */
const POPUP_MAX_W = 260;
/** Smallest gap kept between the popup and the window edge. */
const EDGE = 8;
/** Room the popup needs below the trigger before it is flipped above it. */
const ROOM = 120;

interface Placement {
  left: number;
  width: number;
  top?: number;
  bottom?: number;
}

/**
 * A question-mark circle that reveals a popup on hover/focus.
 *
 * The popup is portalled to the body and positioned in viewport coordinates.
 * Rendered inside the tree it would be clipped by whatever scroll container
 * or pane held it — which is what used to cut the popup off on the wide form
 * columns — and it flips above the trigger when there is no room below.
 */
export function HelpTip({ text }: Props) {
  const trigger = useRef<HTMLSpanElement | null>(null);
  const [placement, setPlacement] = useState<Placement | null>(null);

  const show = () => {
    const element = trigger.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const width = Math.min(POPUP_MAX_W, window.innerWidth - EDGE * 2);
    const centred = rect.left + rect.width / 2 - width / 2;
    const left = Math.max(
      EDGE,
      Math.min(centred, window.innerWidth - width - EDGE),
    );
    const roomBelow = window.innerHeight - rect.bottom;
    if (roomBelow >= ROOM || rect.top < ROOM) {
      setPlacement({ left, width, top: rect.bottom + 6 });
    } else {
      setPlacement({
        left,
        width,
        bottom: window.innerHeight - rect.top + 6,
      });
    }
  };

  const hide = () => setPlacement(null);

  return (
    <span
      ref={trigger}
      className="help-tip"
      tabIndex={0}
      role="button"
      aria-label={text}
      onMouseEnter={show}
      onFocus={show}
      onMouseLeave={hide}
      onBlur={hide}
      onClick={(e) => {
        // Don't toggle an enclosing <label> control when clicked.
        e.preventDefault();
        e.stopPropagation();
      }}
    >
      ?
      {placement !== null &&
        createPortal(
          <span
            className="help-popup"
            role="tooltip"
            style={{
              left: placement.left,
              width: placement.width,
              top: placement.top,
              bottom: placement.bottom,
            }}
          >
            {text}
          </span>,
          document.body,
        )}
    </span>
  );
}
