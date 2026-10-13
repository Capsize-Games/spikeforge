import type { LucideIcon } from "lucide-react";

interface Props {
  /** The Lucide icon rendered at the shared 16px icon size. */
  icon: LucideIcon;
  /** Accessible name; also the tooltip unless `title` overrides it. */
  label: string;
  /** Action for a button; omit when `href` makes the control a link. */
  onClick?: () => void;
  /** When set, the control is an anchor that opens in a new tab. */
  href?: string;
  disabled?: boolean;
  /** Tooltip text when it should differ from the accessible name. */
  title?: string;
  /** Extra classes for a size or colour override. */
  className?: string;
  /** `aria-expanded` for a button that opens a menu. */
  expanded?: boolean;
  /** `aria-haspopup="menu"` for a button that opens a menu. */
  menu?: boolean;
}

/** Icon-only control: 28x28, neutral until hovered. */
export function IconButton({
  icon: Icon,
  label,
  onClick,
  href,
  disabled = false,
  title,
  className,
  expanded,
  menu,
}: Props) {
  const classes = className === undefined ? "icon-btn" : `icon-btn ${className}`;
  const glyph = <Icon size={16} strokeWidth={1.75} aria-hidden="true" />;

  // A link leaves the application, so it opens in a new tab and carries the
  // usual `rel` guard; a button keeps the menu/expanded semantics.
  if (href !== undefined) {
    return (
      <a
        className={classes}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        title={title ?? label}
        aria-label={label}
      >
        {glyph}
      </a>
    );
  }

  return (
    <button
      type="button"
      className={classes}
      onClick={onClick}
      disabled={disabled}
      title={title ?? label}
      aria-label={label}
      aria-expanded={expanded}
      aria-haspopup={menu === true ? "menu" : undefined}
    >
      {glyph}
    </button>
  );
}
