interface Props {
  text: string;
}

/** A question-mark circle that reveals a popup on hover/focus. */
export function HelpTip({ text }: Props) {
  return (
    <span
      className="help-tip"
      tabIndex={0}
      role="button"
      aria-label={text}
      onClick={(e) => {
        // Don't toggle an enclosing <label> control when clicked.
        e.preventDefault();
        e.stopPropagation();
      }}
    >
      ?
      <span className="help-popup" role="tooltip">
        {text}
      </span>
    </span>
  );
}
