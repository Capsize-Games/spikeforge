/** Format a single deployment constraint without hiding its value. */

/** Render a constraint value as text, keeping structured values readable. */
export function formatConstraint(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}
