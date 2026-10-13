import { useI18n } from "../i18n/I18nProvider";
import type {
  DatasetInfo,
  EncodeConfig,
  ModelLoadedPayload,
  SavedModel,
  TrainConfig,
} from "../types";
import { Controls } from "./Controls";
import { DockPane } from "./DockPane";
import { ModelPanel } from "./ModelPanel";
import { NetworkInspector } from "./NetworkInspector";
import { Section } from "./Stepper";
import { TrainControls } from "./TrainControls";

interface Props {
  config: EncodeConfig;
  model: TrainConfig;
  datasets: DatasetInfo[];
  models: SavedModel[];
  /** Name of the loaded checkpoint, or null. */
  currentModel: string | null;
  /** The loaded checkpoint's own record, for the inspector summary. */
  loaded: ModelLoadedPayload | null;
  sample: number[][] | null;
  eventFrame: number[][] | null;
  topologies: string[];
  neurons: string[];
  surrogates: string[];
  gpuAvailable: boolean;
  connected: boolean;
  /** True while a training run is in flight. */
  busy: boolean;
  loading: boolean;
  readOnly: boolean;
  locked: boolean;
  onPatchConfig: (patch: Partial<EncodeConfig>) => void;
  onModelChange: (patch: Partial<TrainConfig>) => void;
  onSelectSample: (patch: Partial<EncodeConfig>) => void;
  onNew: () => void;
  onLoad: (name: string) => void;
  onSave: (name: string) => void;
  onTrain: () => void;
  onStopTrain: () => void;
}

/** Input geometry off a frame the server sent, which is [row][column]. */
function frameDims(frame: number[][] | null): string | null {
  if (frame === null) return null;
  const head = frame[0];
  if (head === undefined || head.length === 0) return null;
  return `${head.length}×${frame.length}`;
}

/**
 * Model & Data as three docked panes: the asset browser, the data and
 * encoding editor, and the network inspector.
 *
 * The widths are the point. The browser and the inspector read badly when
 * stretched, so they take a fixed share and the editor — where the work
 * happens — keeps every remaining pixel. Nothing here is a card: the panes
 * reach the bottom of the window and are separated by one hairline.
 */
export function ModelWorkspace({
  config,
  model,
  datasets,
  models,
  currentModel,
  loaded,
  sample,
  eventFrame,
  topologies,
  neurons,
  surrogates,
  gpuAvailable,
  connected,
  busy,
  loading,
  readOnly,
  locked,
  onPatchConfig,
  onModelChange,
  onSelectSample,
  onNew,
  onLoad,
  onSave,
  onTrain,
  onStopTrain,
}: Props) {
  const { t } = useI18n();
  const selected = datasets.find((d) => d.name === config.dataset);
  const eventMode = selected?.modality === "event";
  const frame = eventMode ? eventFrame : sample;

  return (
    <div className="dock">
      <DockPane variant="assets" heading={t("section.model")}>
        <ModelPanel
          models={models}
          current={currentModel}
          connected={connected}
          busy={busy}
          loading={loading}
          readOnly={readOnly}
          onNew={onNew}
          onLoad={onLoad}
          onSave={onSave}
        />

        <Section title={t("section.train")} hint={t("section.train.hint")}>
          <TrainControls
            running={busy}
            connected={connected}
            readOnly={readOnly}
            onTrain={onTrain}
            onStop={onStopTrain}
          />
        </Section>
      </DockPane>

      <DockPane variant="editor" heading={t("pane.data")} meta={config.dataset}>
        <Controls
          config={config}
          datasets={datasets}
          sample={sample}
          eventFrame={eventFrame}
          locked={locked}
          onChange={onPatchConfig}
          onSelectSample={onSelectSample}
        />
      </DockPane>

      <NetworkInspector
        config={config}
        model={model}
        dataset={selected}
        inputDims={frameDims(frame)}
        loaded={loaded}
        gpuAvailable={gpuAvailable}
        locked={locked}
        topologies={topologies}
        neurons={neurons}
        surrogates={surrogates}
        onModelChange={onModelChange}
      />
    </div>
  );
}
