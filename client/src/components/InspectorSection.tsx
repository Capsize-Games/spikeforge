import type { ReactNode } from "react";

interface Props {
  title: string;
  children: ReactNode;
  className?: string;
}

/** A titled group in the inspector column. */
export function InspectorSection({ title, children, className }: Props) {
  return (
    <section
      className={
        className === undefined
          ? "inspector-section"
          : `inspector-section ${className}`
      }
    >
      <h2 className="inspector-section-title">{title}</h2>
      {children}
    </section>
  );
}
