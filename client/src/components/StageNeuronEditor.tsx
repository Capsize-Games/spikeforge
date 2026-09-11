import { useStageNeurons } from "../hooks/useStageNeurons";
import type { StageNeurons } from "../types";

interface Props {
  /** Per-stage neuron-kind overrides keyed by stage name. */
  value: StageNeurons;
  /** Registered neuron kinds offered by the picker. */
  neurons: string[];
  disabled: boolean;
  onChange: (value: StageNeurons) => void;
}

/** Per-stage neuron-kind overrides for the topology panel. */
export function StageNeuronEditor({
  value,
  neurons,
  disabled,
  onChange,
}: Props) {
  const fallback = neurons[0] ?? "leaky";
  const { draft, setDraft, submit } = useStageNeurons((stage) =>
    onChange({ ...value, [stage]: fallback }),
  );

  const remove = (stage: string) => {
    const next = { ...value };
    delete next[stage];
    onChange(next);
  };

  const kinds = (current: string) => (neurons.length ? neurons : [current]);

  return (
    <div className="stage-neurons">
      {Object.entries(value).map(([stage, kind]) => (
        <div className="model-row" key={stage}>
          <span className="muted">{stage}</span>
          <select
            className="text-input"
            value={kind}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, [stage]: e.target.value })}
          >
            {kinds(kind).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="apply small"
            disabled={disabled}
            onClick={() => remove(stage)}
          >
            ✕
          </button>
        </div>
      ))}
      <div className="model-row">
        <input
          className="text-input"
          placeholder="stage name (e.g. lif1)"
          value={draft}
          disabled={disabled}
          onChange={(e) => setDraft(e.target.value)}
        />
        <button
          type="button"
          className="apply small"
          disabled={disabled || !draft.trim()}
          onClick={submit}
        >
          Add stage
        </button>
      </div>
    </div>
  );
}
