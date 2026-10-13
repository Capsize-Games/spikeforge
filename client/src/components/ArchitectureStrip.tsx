import { useI18n } from "../i18n/I18nProvider";
import type { TranslationKey } from "../i18n/translations";
import type { DatasetInfo, EncodeConfig, TrainConfig } from "../types";
import { FactGrid } from "./FactGrid";
import type { Fact } from "./FactGrid";

interface Props {
  config: EncodeConfig;
  model: TrainConfig;
  /** The configured dataset, once the server has listed it. */
  dataset: DatasetInfo | undefined;
  /** Input geometry read off the last frame the server sent, e.g. "28×28". */
  inputDims: string | null;
  gpuAvailable: boolean;
}

/** Where the coding label comes from; the server names the coding type. */
const CODING_KEYS: Record<EncodeConfig["coding"], TranslationKey> = {
  rate: "coding.rate",
  latency: "coding.latency",
  delta: "coding.delta",
  random: "coding.random",
};

/** One stage of the chain the product builds. */
interface Stage {
  key: string;
  label: string;
  value: string;
  unit?: string;
}

/**
 * The architecture the application is actually configured to build:
 * Input → Encoder → Hidden → Output, with the values it reads for each.
 *
 * A stage whose value the application does not hold shows a dash rather than
 * a guess; there is no parameter count or benchmark here because nothing in
 * the client's state can produce one.
 */
export function ArchitectureStrip({
  config,
  model,
  dataset,
  inputDims,
  gpuAvailable,
}: Props) {
  const { t } = useI18n();

  const stages: Stage[] = [
    { key: "input", label: t("arch.input"), value: inputDims ?? "—" },
    {
      key: "encoder",
      label: t("arch.encoder"),
      value: t(CODING_KEYS[config.coding]),
    },
    {
      key: "hidden",
      label: t("arch.hidden"),
      value: String(model.hidden),
      unit: t("arch.neurons"),
    },
    {
      key: "output",
      label: t("arch.output"),
      value: dataset === undefined ? "—" : String(dataset.classes),
      unit: t("arch.classes"),
    },
  ];

  const facts: Fact[] = [
    { label: t("field.dataset"), value: dataset?.name ?? config.dataset },
    { label: "num_steps", value: String(config.num_steps) },
    {
      label: t("arch.device"),
      value: `${model.device.toUpperCase()} · ${
        gpuAvailable ? t("arch.gpuAvailable") : t("arch.gpuUnavailable")
      }`,
    },
  ];

  return (
    <div className="arch-summary">
      <div className="section-micro">{t("arch.title")}</div>
      <ol className="arch-strip">
        {stages.map((stage) => (
          <li className="arch-node" key={stage.key}>
            <span className="arch-node-label">{stage.label}</span>
            <span className="arch-node-value">{stage.value}</span>
            {stage.unit !== undefined && (
              <span className="arch-node-unit">{stage.unit}</span>
            )}
          </li>
        ))}
      </ol>
      <FactGrid facts={facts} />
    </div>
  );
}
