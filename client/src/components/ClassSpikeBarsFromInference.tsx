import type { InferencePayload } from "../types";
import { ClassSpikeBars } from "./ClassSpikeBars";

interface Props {
  inference: InferencePayload;
  timeStep?: number | null;
}

/** Build the class bars straight from an inference payload. */
export function ClassSpikeBarsFromInference({
  inference,
  timeStep = null,
}: Props) {
  return (
    <ClassSpikeBars
      classSpikes={inference.class_spikes}
      predicted={inference.predicted}
      trueLabel={inference.true_label}
      outputOverTime={inference.output_over_time}
      timeStep={timeStep}
    />
  );
}
