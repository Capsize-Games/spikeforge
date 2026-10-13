/**
 * The dataset picker's options and the dataset facts, built from what the
 * server listed.
 *
 * The names and counts are the server's own; nothing here is estimated or
 * filled in when the list has not arrived.
 */

import type { TranslationKey } from "../i18n/translations";
import type { DatasetInfo } from "../types";
import type { Fact } from "./FactGrid";
import type { Option } from "./SelectField";

type Translate = (key: TranslationKey) => string;

/** Human label for a dataset option, including modality and availability. */
export function datasetLabel(d: DatasetInfo, t: Translate): string {
  const base = `${d.name} (${d.classes} ${t("dataset.classes")})`;
  if (d.modality !== "event") return base;
  if (d.available === false) {
    return `${base} — ${t("dataset.unavailable")}`;
  }
  return `${base} — ${t("dataset.events")}`;
}

/** Build the dataset dropdown, disabling event sets with no tonic loader. */
export function datasetOptions(
  datasets: DatasetInfo[],
  current: string,
  t: Translate,
): Option[] {
  if (datasets.length === 0) return [{ value: current, label: current }];
  return datasets.map((d) => ({
    value: d.name,
    label: datasetLabel(d, t),
    // Unavailable events would otherwise fail inside the download worker.
    disabled: d.modality === "event" && d.available === false,
  }));
}

/** The properties of the configured dataset, or none when it is unlisted. */
export function datasetFacts(
  dataset: DatasetInfo | undefined,
  t: Translate,
): Fact[] {
  if (dataset === undefined) return [];
  return [
    {
      label: t("dataset.modality"),
      value:
        dataset.modality === "event"
          ? t("modality.event")
          : t("modality.image"),
    },
    { label: t("dataset.classes"), value: String(dataset.classes) },
    {
      label: t("dataset.availability"),
      value:
        dataset.available === false
          ? t("dataset.unavailable")
          : t("dataset.available"),
    },
  ];
}
