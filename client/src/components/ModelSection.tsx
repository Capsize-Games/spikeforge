import { TRAIN_HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import type { TopologyParams, TrainConfig } from "../types";
import { HelpTip } from "./HelpTip";
import { SelectField } from "./SelectField";
import { NumberField } from "./NumberField";
import { StageNeuronEditor } from "./StageNeuronEditor";
import { Section } from "./Stepper";

interface Props {
  model: TrainConfig;
  gpuAvailable: boolean;
  /** True when a checkpoint is loaded: architecture controls are read-only. */
  locked: boolean;
  topologies: string[];
  neurons: string[];
  surrogates: string[];
  onChange: (patch: Partial<TrainConfig>) => void;
}

/** Registry default used when no neuron override is stored. */
const DEFAULT_NEURON = "leaky";

/** Read a string topology override, ignoring non-string values. */
function stringParam(params: TopologyParams, key: string): string | null {
  const value = params[key];
  return typeof value === "string" ? value : null;
}

/** Merge one topology override; a null value drops the key entirely. */
function withParam(
  params: TopologyParams,
  key: string,
  value: string | null,
): TopologyParams {
  const next = { ...params };
  if (value === null) {
    delete next[key];
  } else {
    next[key] = value;
  }
  return next;
}

/** Fall back to the current value so a picker never renders empty. */
function names(values: string[], current: string): string[] {
  if (values.length > 0) return values;
  return current ? [current] : [];
}

/** Architecture pickers and numeric model settings for the Model section. */
export function ModelSection({
  model,
  gpuAvailable,
  locked,
  topologies,
  neurons,
  surrogates,
  onChange,
}: Props) {
  const { t } = useI18n();
  const set = (patch: Partial<TrainConfig>) => onChange(patch);
  const params = model.topology_params;
  const neuron = stringParam(params, "neuron") ?? DEFAULT_NEURON;
  const surrogate = stringParam(params, "surrogate") ?? "";

  // The current value is always present so an unloaded list cannot blank out.
  const topologyOptions = names(topologies, model.topology).map((v) => ({
    value: v,
    label: v,
  }));
  const neuronOptions = names(neurons, neuron).map((v) => ({
    value: v,
    label: v,
  }));
  const surrogateOptions = [
    { value: "", label: t("model.default") },
    ...names(surrogates, surrogate).map((v) => ({ value: v, label: v })),
  ];

  return (
    <Section title={t("section.model")} hint={t("section.model.hint")}>
      <SelectField
        label={t("field.topology")}
        value={model.topology}
        help={TRAIN_HELP.topology}
        tour="topology"
        disabled={locked}
        options={topologyOptions}
        onChange={(v) => set({ topology: v })}
      />
      <SelectField
        label={t("field.neuron")}
        value={neuron}
        help={TRAIN_HELP.neuron}
        tour="neuron"
        disabled={locked}
        options={neuronOptions}
        onChange={(v) =>
          set({ topology_params: withParam(params, "neuron", v) })
        }
      />
      <SelectField
        label={t("field.surrogate")}
        value={surrogate}
        help={TRAIN_HELP.surrogate}
        tour="surrogate"
        disabled={locked}
        options={surrogateOptions}
        onChange={(v) =>
          set({ topology_params: withParam(params, "surrogate", v || null) })
        }
      />
      <p className="panel-note">
        {t("model.stageOverrides")}
        <HelpTip text={TRAIN_HELP.stage_neurons} />
      </p>
      <StageNeuronEditor
        value={model.stage_neurons}
        neurons={neurons}
        disabled={locked}
        onChange={(v) => set({ stage_neurons: v })}
      />

      <NumberField
        label="hidden"
        value={model.hidden}
        min={16}
        max={512}
        step={16}
        help={TRAIN_HELP.hidden}
        disabled={locked}
        onChange={(v) => set({ hidden: v })}
      />
      <NumberField
        label="beta"
        value={model.beta}
        min={0.1}
        max={0.95}
        step={0.05}
        range
        help={TRAIN_HELP.beta}
        disabled={locked}
        onChange={(v) => set({ beta: v })}
      />
      <NumberField
        label="lr"
        value={model.lr}
        min={0.001}
        max={0.05}
        step={0.001}
        help={TRAIN_HELP.lr}
        onChange={(v) => set({ lr: v })}
      />
      <NumberField
        label="epochs"
        value={model.epochs}
        min={1}
        max={10}
        step={1}
        help={TRAIN_HELP.epochs}
        onChange={(v) => set({ epochs: v })}
      />

      <SelectField
        label={t("field.device")}
        value={model.device}
        help={TRAIN_HELP.device}
        options={[
          { value: "auto", label: t("device.auto") },
          {
            value: "gpu",
            label: gpuAvailable ? "GPU" : t("device.unavailable"),
            disabled: !gpuAvailable,
          },
          { value: "cpu", label: "CPU" },
        ]}
        onChange={(v) => set({ device: v as TrainConfig["device"] })}
      />

      {!locked && <p className="arch-note">{t("model.runNote")}</p>}
    </Section>
  );
}
