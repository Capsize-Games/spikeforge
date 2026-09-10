import type {
  DeploymentNodes,
  SubstitutionRecord,
} from "../targetTypes";

interface Props {
  nodes: DeploymentNodes;
}

/** Format one substitution as "name (primitive→substitute)". */
function subLine(item: SubstitutionRecord): string {
  return `${item.name} (${item.primitive}→${item.substitute})`;
}

/** One labelled node bucket, honest about whether it is empty. */
interface Bucket {
  label: string;
  names: string[];
  tone: string;
}

/** The supported/substituted/unsupported node buckets of a report. */
export function DeploymentBuckets({ nodes }: Props) {
  const buckets: Bucket[] = [
    { label: "Supported", names: nodes.supported, tone: "supported" },
    {
      label: "Substituted",
      names: nodes.substituted.map(subLine),
      tone: "substituted",
    },
    { label: "Unsupported", names: nodes.unsupported, tone: "unsupported" },
  ];

  return (
    <div className="deployment-buckets">
      {buckets.map((bucket) => (
        <div className={`bucket ${bucket.tone}`} key={bucket.label}>
          <div className="bucket-head">
            {bucket.label} · {bucket.names.length}
          </div>
          {bucket.names.length === 0 ? (
            <div className="bucket-empty">none</div>
          ) : (
            <div className="bucket-names">{bucket.names.join(", ")}</div>
          )}
        </div>
      ))}
    </div>
  );
}
