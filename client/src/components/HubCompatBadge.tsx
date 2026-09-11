import type { HubVerdict } from "../hubTypes";

interface Props {
  verdict: HubVerdict;
}

/** A small badge naming a compatibility verdict. */
export function HubCompatBadge({ verdict }: Props) {
  return <span className={`hub-verdict ${verdict}`}>{verdict}</span>;
}
