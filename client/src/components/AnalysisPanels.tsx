import type {
  BenchmarkPayload,
  EncodingReportPayload,
  SurrogateCurvePayload,
  TrajectoryMetricsPayload,
} from "../introspectionTypes";
import type { ExecutionMode } from "../types";
import { BenchmarkPanel } from "./BenchmarkPanel";
import { EncodingReportPanel } from "./EncodingReportPanel";
import { MetricsPanel } from "./MetricsPanel";
import { SurrogateCurvePanel } from "./SurrogateCurvePanel";

interface Props {
  mode: ExecutionMode;
  metrics: TrajectoryMetricsPayload | null;
  encodingReport: EncodingReportPayload | null;
  surrogateCurve: SurrogateCurvePayload | null;
  benchmark: BenchmarkPayload | null;
  benchmarkLoading: boolean;
  surrogates: string[];
  currentSurrogate: string;
  onRefreshMetrics: () => void;
  onRefreshEncodingReport: () => void;
  onRequestSurrogateCurve: (name: string) => void;
  onRunBenchmark: () => void;
  onSwitchMode: () => void;
}

/** The RIGHT-column analysis stack: metrics, encoding, surrogate, bench. */
export function AnalysisPanels({
  mode,
  metrics,
  encodingReport,
  surrogateCurve,
  benchmark,
  benchmarkLoading,
  surrogates,
  currentSurrogate,
  onRefreshMetrics,
  onRefreshEncodingReport,
  onRequestSurrogateCurve,
  onRunBenchmark,
  onSwitchMode,
}: Props) {
  return (
    <div className="analysis-grid">
      <MetricsPanel
        mode={mode}
        metrics={metrics}
        onRefresh={onRefreshMetrics}
        onSwitchMode={onSwitchMode}
      />
      <EncodingReportPanel
        report={encodingReport}
        onRefresh={onRefreshEncodingReport}
      />
      <SurrogateCurvePanel
        names={surrogates}
        current={currentSurrogate}
        curve={surrogateCurve}
        onRequest={onRequestSurrogateCurve}
      />
      <BenchmarkPanel
        payload={benchmark}
        loading={benchmarkLoading}
        onRun={onRunBenchmark}
      />
    </div>
  );
}
