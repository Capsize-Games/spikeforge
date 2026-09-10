/** Formatting helpers for the benchmark readout. */

import type { BenchmarkMemory } from "../introspectionTypes";

/** Fixed-precision milliseconds, or an em dash when unavailable. */
export function formatMs(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return value < 1 ? value.toFixed(3) : value.toFixed(2);
}

/** Steps per second, rounded for a compact cell. */
export function formatRate(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

const UNITS = ["B", "KB", "MB", "GB"] as const;

/** Human byte size, or an em dash for a null measurement. */
export function formatBytes(value: number): string {
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < UNITS.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || size >= 10 ? 0 : 1;
  return `${size.toFixed(digits)} ${UNITS[unit]}`;
}

/** One benchmark memory cell: the value plus the measurement it came from. */
export interface MemoryCell {
  text: string;
  source: string;
}

/** Pick the best available peak-memory measurement, labelled honestly. */
export function memoryCell(memory: BenchmarkMemory): MemoryCell {
  if (memory.cuda_peak_bytes !== null) {
    return { text: formatBytes(memory.cuda_peak_bytes), source: "CUDA peak" };
  }
  if (memory.process_rss_bytes !== null) {
    return {
      text: formatBytes(memory.process_rss_bytes),
      source: "process RSS",
    };
  }
  if (memory.tracemalloc_peak_bytes !== null) {
    return {
      text: formatBytes(memory.tracemalloc_peak_bytes),
      source: "tracemalloc peak",
    };
  }
  return { text: "—", source: "no memory measurement available" };
}
