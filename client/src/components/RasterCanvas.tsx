import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import type { RasterPayload } from "../types";

interface Props {
  raster: RasterPayload | null;
  width?: number;
  height?: number;
  label?: string;
  /** Right-aligned controls rendered in the panel header. */
  actions?: ReactNode;
  /** Time step to spotlight, e.g. while the spike-frame animation plays. */
  highlightStep?: number | null;
  /** Shown centred when there is no raster yet. */
  emptyNote?: string;
  /** Label each neuron row (e.g. class digits) when the layer is small. */
  yLabels?: string[];
  /** Rows to shade, e.g. the predicted and true classes. */
  highlightRows?: number[];
  /** Reorder neurons by firing rate so silent / hot units stand out. */
  sortByRate?: boolean;
  /** Small caption rendered under the canvas. */
  summary?: string;
}

/** Neuron draw order on the y-axis (top = most active when sorted). */
function rowOrder(raster: RasterPayload, sortByRate: boolean): number[] {
  const order = Array.from({ length: max1(raster.num_neurons) }, (_, i) => i);
  if (!sortByRate) return order;
  const counts = new Array(max1(raster.num_neurons)).fill(0);
  raster.neurons.forEach((n) => {
    counts[n] += 1;
  });
  return order.slice().sort((a, b) => counts[b] - counts[a]);
}

function max1(n: number): number {
  return Math.max(1, n);
}

/** Pixel y for a neuron/rank on the plot (0 at the bottom). */
function yFor(pos: number, neurons: number, plotH: number): number {
  return 8 + plotH - (pos / neurons) * plotH;
}

export function RasterCanvas({
  raster,
  width = 600,
  height = 260,
  label,
  actions,
  highlightStep = null,
  emptyNote,
  yLabels,
  highlightRows,
  sortByRate = false,
  summary,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [plotWidth, setPlotWidth] = useState(width);

  // Track the panel width so canvas pixels map 1:1 to display pixels,
  // which keeps the time axis aligned with the Time cursor slider.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const cw = Math.floor(entries[0].contentRect.width);
      if (cw > 0) setPlotWidth(cw);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!raster) {
      if (emptyNote) {
        ctx.fillStyle = "#8b949e";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(emptyNote, canvas.width / 2, canvas.height / 2);
        ctx.textAlign = "start";
      }
      return;
    }

    const steps = max1(raster.num_steps);
    const neurons = max1(raster.num_neurons);
    const padL = 0;
    const padR = 0;
    const padB = 22;
    const plotW = canvas.width - padL - padR;
    const plotH = canvas.height - padB - 14;
    const bandH = plotH / neurons;

    ctx.strokeStyle = "#30363d";
    ctx.strokeRect(padL, 8, plotW, plotH);

    const order = rowOrder(raster, sortByRate);
    const rank = new Array(neurons).fill(0);
    order.forEach((neuron, pos) => {
      rank[neuron] = pos;
    });

    (highlightRows ?? []).forEach((row) => {
      if (row < 0 || row >= neurons) return;
      ctx.fillStyle = "rgba(63, 185, 80, 0.14)";
      ctx.fillRect(padL, yFor(rank[row], neurons, plotH) - bandH / 2, plotW, bandH);
    });

    ctx.fillStyle = "#e3b341";
    for (let i = 0; i < raster.time.length; i++) {
      const x = padL + (raster.time[i] / steps) * plotW;
      const y = yFor(rank[raster.neurons[i]] ?? 0, neurons, plotH);
      ctx.fillRect(x, y, 1.4, 1.4);
    }

    if (highlightStep !== null && highlightStep !== undefined) {
      const idx = Math.max(0, Math.min(highlightStep, steps - 1));
      const x0 = padL + (idx / steps) * plotW;
      const bandW = Math.max(2, plotW / steps);
      ctx.fillStyle = "rgba(88, 166, 255, 0.16)";
      ctx.fillRect(x0, 8, bandW, plotH);
      ctx.fillStyle = "#79c0ff";
      for (let i = 0; i < raster.time.length; i++) {
        if (raster.time[i] !== idx) continue;
        const x = padL + (raster.time[i] / steps) * plotW;
        const y = yFor(rank[raster.neurons[i]] ?? 0, neurons, plotH);
        ctx.fillRect(x, y, 2.4, 2.4);
      }
      ctx.strokeStyle = "rgba(88, 166, 255, 0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + bandW / 2, 8);
      ctx.lineTo(x0 + bandW / 2, 8 + plotH);
      ctx.stroke();
    }

    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#8b949e";
    ctx.fillText("Time step", plotW / 2 - 18, canvas.height - 6);
    ctx.fillText(String(steps), plotW - 20, canvas.height - 6);
  }, [
    raster,
    width,
    height,
    highlightStep,
    emptyNote,
    yLabels,
    highlightRows,
    sortByRate,
  ]);

  // Rail labels mirror the canvas row geometry so they line up with dots.
  const rowTop = 8;
  const plotHeight = Math.max(1, height - 22 - 14);
  const rowCount = raster ? max1(raster.num_neurons) : 0;
  const railLabels =
    raster && yLabels && yLabels.length === rowCount && rowCount <= 16
      ? rowOrder(raster, sortByRate).map((neuron, pos) => ({
          key: neuron,
          text: yLabels[neuron] ?? "",
          top: rowTop + plotHeight - (pos / rowCount) * plotHeight - 6,
        }))
      : [];

  return (
    <div className="panel">
      {(label || actions) && (
        <div className="panel-title row-title">
          {label && <span>{label}</span>}
          {actions && <span className="panel-actions">{actions}</span>}
        </div>
      )}
      <div className="raster-body">
        <div className="raster-rail" style={{ height }}>
          {railLabels.map(({ key, text, top }) => (
            <span key={key} className="rail-label" style={{ top }}>
              {text}
            </span>
          ))}
        </div>
        <div className="raster-wrap" ref={wrapRef}>
          <canvas ref={canvasRef} width={plotWidth} height={height} />
        </div>
      </div>
      {summary && <div className="raster-summary">{summary}</div>}
    </div>
  );
}
