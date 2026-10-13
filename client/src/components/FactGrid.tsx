/** One label/value pair of a read-out. */
export interface Fact {
  label: string;
  value: string;
}

/**
 * A compact label/value grid for facts the application actually holds.
 *
 * Every label is one of the product's own parameter names or an interface
 * string; a fact the application does not know is left out of the array
 * rather than shown as a placeholder.
 */
export function FactGrid({ facts }: { facts: Fact[] }) {
  if (facts.length === 0) return null;
  return (
    <dl className="fact-grid">
      {facts.map((fact) => (
        <div className="fact" key={fact.label}>
          <dt>{fact.label}</dt>
          <dd>{fact.value}</dd>
        </div>
      ))}
    </dl>
  );
}
