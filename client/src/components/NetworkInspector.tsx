import { useI18n } from "../i18n/I18nProvider";
import type {
  DatasetInfo,
  EncodeConfig,
  ModelLoadedPayload,
  TrainConfig,
} from "../types";
import { ArchitectureStrip } from "./ArchitectureStrip";
import { DockPane } from "./DockPane";
import { LoadedModelPanel } from "./LoadedModelPanel";
import { ModelSection } from "./ModelSection";

interface Props {
  config: EncodeConfig;
  model: TrainConfig;
  /** The configured dataset, once the server has listed it. */
  dataset: DatasetInfo | undefined;
  /** Input geometry read off the last frame the server sent. */
  inputDims: string | null;
  /** The loaded checkpoint, whose own record is summarised here. */
  loaded: ModelLoadedPayload | null;
  gpuAvailable: boolean;
  /** True when a checkpoint is loaded: architecture controls are read-only. */
  locked: boolean;
  topologies: string[];
  neurons: string[];
  surrogates: string[];
  onModelChange: (patch: Partial<TrainConfig>) => void;
}

/**
 * The network inspector pane: what the configured network is, and every
 * setting that shapes it.
 *
 * The architecture strip comes first because it is the one thing to read at a
 * glance, then the loaded checkpoint's own record when there is one, then the
 * editors — the same controls they have always been, with the parameters that
 * need an exact value drawn as boxes.
 */
export function NetworkInspector({
  config,
  model,
  dataset,
  inputDims,
  loaded,
  gpuAvailable,
  locked,
  topologies,
  neurons,
  surrogates,
  onModelChange,
}: Props) {
  const { t } = useI18n();
  return (
    <DockPane
      variant="inspector"
      heading={t("pane.network")}
      meta={model.topology}
    >
      <ArchitectureStrip
        config={config}
        model={model}
        dataset={dataset}
        inputDims={inputDims}
        gpuAvailable={gpuAvailable}
      />

      {loaded !== null && <LoadedModelPanel loaded={loaded} />}

      <ModelSection
        model={model}
        gpuAvailable={gpuAvailable}
        locked={locked}
        topologies={topologies}
        neurons={neurons}
        surrogates={surrogates}
        onChange={onModelChange}
      />
    </DockPane>
  );
}
