import type {
  BenchmarkPayload,
  BenchmarkRecord,
} from "../introspectionTypes";
import { HELP } from "../helpText";
import { formatMs, formatRate, memoryCell } from "./benchmarkFormat";
import { HelpTip } from "./HelpTip";

interface Props {
  payload: BenchmarkPayload | null;
  loading: boolean;
  onRun: () => void;
}

/** Environment and resolved-fixture summary above the results table. */
function EnvLine({ payload }: { payload: BenchmarkPayload }) {
  const env = payload.environment;
  const cfg = payload.config;
  return (
    <>
      <div className="panel-caption">
        torch {env.torch_version} · {env.device} · cuda{" "}
        {env.cuda_available ? "yes" : "no"} · compile{" "}
        {env.compile_available ? "yes" : "no"}
      </div>
      <div className="panel-caption">
        {cfg.topologies.join(", ")} · {cfg.steps} steps · batch{" "}
        {cfg.batch_size} · {cfg.repeats} repeats
        {cfg.backward ? " · backward" : ""}
      </div>
    </>
  );
}

/** One topology/mode benchmark row. */
function Row({ record }: { record: BenchmarkRecord }) {
  const mem = memoryCell(record.memory);
  return (
    <tr>
      <td className="readout-name" title={record.topology}>
        {record.topology}
      </td>
      <td>{record.mode}</td>
      <td>{record.compiled ? "yes" : "no"}</td>
      <td title={record.compile_status}>{record.compile_status}</td>
      <td>{formatMs(record.forward.mean_ms_per_step)}</td>
      <td>{formatRate(record.forward.steps_per_second)}</td>
      <td title={mem.source}>{mem.text}</td>
    </tr>
  );
}

/** Compact benchmark readout with an explicit, user-triggered run. */
export function BenchmarkPanel({ payload, loading, onRun }: Props) {
  return (
    <div className="panel benchmark-panel" data-tour="benchmark">
      <div className="panel-title row-title">
        <span>
          Benchmark
          <HelpTip text={HELP.benchmark} />
        </span>
        <span className="panel-actions">
          <button
            type="button"
            className="apply small"
            onClick={onRun}
            disabled={loading}
          >
            {loading ? "Running…" : "Run benchmark"}
          </button>
        </span>
      </div>

      {loading && (
        <div className="panel-note">Running benchmark on a tiny fixture…</div>
      )}

      {!loading && payload === null && (
        <div className="panel-note">
          No benchmark run yet. Press Run benchmark to time this machine.
        </div>
      )}

      {payload !== null && (
        <>
          <EnvLine payload={payload} />
          <div className="table-scroll">
            <table className="readout-table">
              <thead>
                <tr>
                  <th>topology</th>
                  <th>mode</th>
                  <th title={HELP.benchmark_compiled}>compiled</th>
                  <th title={HELP.benchmark_compiled}>status</th>
                  <th>ms/step</th>
                  <th>steps/s</th>
                  <th title={HELP.benchmark_memory}>peak mem</th>
                </tr>
              </thead>
              <tbody>
                {payload.results.map((record) => (
                  <Row
                    key={`${record.topology}-${record.mode}`}
                    record={record}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
