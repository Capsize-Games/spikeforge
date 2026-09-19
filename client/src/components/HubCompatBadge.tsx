import type { HubVerdict } from "../hubTypes";
import { Badge } from "./Badge";
import type { BadgeTone } from "./Badge";

const TONES: Record<HubVerdict, BadgeTone> = {
  exact: "ok",
  mappable: "warn",
  incompatible: "bad",
};

interface Props {
  verdict: HubVerdict;
}

/** A small badge naming a compatibility verdict. */
export function HubCompatBadge({ verdict }: Props) {
  return <Badge tone={TONES[verdict]}>{verdict}</Badge>;
}
