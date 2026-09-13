import { HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import type { EncodeConfig } from "../types";
import { CheckField } from "./CheckField";
import { InputSizeField } from "./InputSizeField";
import { SelectField } from "./SelectField";
import { SliderField } from "./SliderField";
import { Section } from "./Stepper";

interface Props {
  config: EncodeConfig;
  /** True when a checkpoint is loaded: encoding is read-only. */
  locked: boolean;
  /** True for event datasets, whose samples are already spike trains. */
  eventMode: boolean;
  onChange: (patch: Partial<EncodeConfig>) => void;
}

/**
 * The Encoding section: the coding picker plus its per-coding parameters.
 *
 * Event datasets carry their own spikes and time bins, so every coding
 * control is disabled with an explicit note rather than being silently
 * ignored; only the playback interval stays adjustable.
 */
export function EncodingControls({
  config,
  locked,
  eventMode,
  onChange,
}: Props) {
  const { t } = useI18n();
  const set = (patch: Partial<EncodeConfig>) => onChange(patch);
  const fixed = locked || eventMode;

  return (
    <Section
      title={t("section.encoding")}
      hint={
        eventMode
          ? t("section.encoding.eventHint")
          : t("section.encoding.imageHint")
      }
    >
      {eventMode && <p className="modality-note">{HELP.event_modal}</p>}

      <SelectField
        label={t("field.coding")}
        value={config.coding}
        help={HELP.coding}
        tour="coding"
        disabled={fixed}
        options={[
          { value: "rate", label: t("coding.rate") },
          { value: "latency", label: t("coding.latency") },
          { value: "delta", label: t("coding.delta") },
          { value: "random", label: t("coding.random") },
        ]}
        onChange={(v) => set({ coding: v as EncodeConfig["coding"] })}
      />

      <SliderField
        label="num_steps"
        value={config.num_steps}
        min={5}
        max={200}
        step={5}
        help={eventMode ? HELP.event_steps : HELP.num_steps}
        disabled={fixed}
        onChange={(v) => set({ num_steps: v })}
      />
      <SliderField
        label={t("field.interval")}
        value={config.interval_ms}
        min={20}
        max={400}
        step={10}
        help={HELP.interval_ms}
        onChange={(v) => set({ interval_ms: v })}
      />
      <InputSizeField
        value={config.input_size}
        disabled={fixed}
        onChange={(v) => set({ input_size: v })}
      />
      <CheckField
        label={t("field.animateHidden")}
        checked={config.animate_hidden}
        help={HELP.animate_hidden}
        disabled={locked}
        onChange={(v) => set({ animate_hidden: v })}
      />

      {!eventMode && config.coding === "rate" && (
        <SliderField
          label="gain"
          value={config.gain}
          min={0.05}
          max={1}
          step={0.05}
          help={HELP.gain}
          disabled={locked}
          onChange={(v) => set({ gain: v })}
        />
      )}

      {!eventMode && config.coding === "latency" && (
        <>
          <SliderField
            label="tau"
            value={config.tau}
            min={1}
            max={20}
            step={0.5}
            help={HELP.tau}
            disabled={locked}
            onChange={(v) => set({ tau: v })}
          />
          <SliderField
            label="threshold"
            value={config.threshold}
            min={0.005}
            max={0.1}
            step={0.005}
            help={HELP.threshold}
            disabled={locked}
            onChange={(v) => set({ threshold: v })}
          />
          <CheckField
            label="linear"
            checked={config.linear}
            disabled={locked}
            help={HELP.linear}
            onChange={(v) => set({ linear: v })}
          />
          <CheckField
            label="normalize"
            checked={config.normalize}
            disabled={locked}
            help={HELP.normalize}
            onChange={(v) => set({ normalize: v })}
          />
          <CheckField
            label="clip"
            checked={config.clip}
            disabled={locked}
            help={HELP.clip}
            onChange={(v) => set({ clip: v })}
          />
        </>
      )}

      {!eventMode && config.coding === "delta" && (
        <SliderField
          label="delta_threshold"
          value={config.delta_threshold}
          min={1}
          max={10}
          step={0.5}
          help={HELP.delta_threshold}
          disabled={locked}
          onChange={(v) => set({ delta_threshold: v })}
        />
      )}

      {!eventMode && config.coding === "random" && (
        <SliderField
          label="random_scale"
          value={config.random_scale}
          min={0.1}
          max={1}
          step={0.05}
          help={HELP.random_scale}
          onChange={(v) => set({ random_scale: v })}
        />
      )}
    </Section>
  );
}
