import { useEffect } from "react";
import type { RefObject } from "react";

import type { CanvasColors } from "../theme";
import type { RasterPayload } from "../types";

/** Neuron draw order on the y-axis (top = most active when sorted). */
export function rowOrder(
  raster: RasterPayload,
  sortByRate: boolean,
): number[] {
  const order = Array.from({ length: max1(raster.num_neurons) }, (_, i) => i);
  if (!sortByRate) return order;
  const counts = new Array(max1(raster.num_neurons)).fill(0);
  raster.neurons.forEach((n) => {
    counts[n] += 1;
  });
  return order.slice().sort((a, b) => counts[b] - counts[a]);
}

export function max1(n: number): number {
  return Math.max(1, n);
}

/** Pixel y for a neuron/rank on the plot (0 at the bottom). */
function yFor(pos: number, neurons: number, plotH: number): number {
  return 8 + plotH - (pos / neurons) * plotH;
}

interface PaintOptions {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  raster: RasterPayload | null;
  /** Real pixel size; rewriting it clears the bitmap, so repaint on change. */
  plotWidth: number;
  measuredHeight: number;
  highlightStep: number | null;
  emptyNote: string | undefined;
  highlightRows: number[] | undefined;
  sortByRate: boolean;
  colors: CanvasColors;
}

/** Paint the raster dots, row shading, and step cursor onto the canvas. */
export function usePaintRaster(options: PaintOptions) {
  const {
    canvasRef,
    raster,
    plotWidth,
    measuredHeight,
    highlightStep,
    emptyNote,
    highlightRows,
    sortByRate,
    colors,
  } = options;

  useEffect(() => {
    // Read the live ref: it is set before effects run on mount, while a
    // render-time element would still be null on the first paint.
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!raster) {
      if (emptyNote) {
        ctx.fillStyle = colors.text;
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

    ctx.strokeStyle = colors.grid;
    ctx.strokeRect(padL, 8, plotW, plotH);

    const order = rowOrder(raster, sortByRate);
    const rank = new Array(neurons).fill(0);
    order.forEach((neuron, pos) => {
      rank[neuron] = pos;
    });

    (highlightRows ?? []).forEach((row) => {
      if (row < 0 || row >= neurons) return;
      ctx.fillStyle = colors.highlight;
      ctx.fillRect(
        padL,
        yFor(rank[row], neurons, plotH) - bandH / 2,
        plotW,
        bandH,
      );
    });

    ctx.fillStyle = colors.dot;
    for (let i = 0; i < raster.time.length; i++) {
      const x = padL + (raster.time[i] / steps) * plotW;
      const y = yFor(rank[raster.neurons[i]] ?? 0, neurons, plotH);
      ctx.fillRect(x, y, 1.4, 1.4);
    }

    if (highlightStep !== null && highlightStep !== undefined) {
      const idx = Math.max(0, Math.min(highlightStep, steps - 1));
      const x0 = padL + (idx / steps) * plotW;
      const bandW = Math.max(2, plotW / steps);
      ctx.fillStyle = colors.accentSoft;
      ctx.fillRect(x0, 8, bandW, plotH);
      ctx.fillStyle = colors.accent;
      for (let i = 0; i < raster.time.length; i++) {
        if (raster.time[i] !== idx) continue;
        const x = padL + (raster.time[i] / steps) * plotW;
        const y = yFor(rank[raster.neurons[i]] ?? 0, neurons, plotH);
        ctx.fillRect(x, y, 2.4, 2.4);
      }
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + bandW / 2, 8);
      ctx.lineTo(x0 + bandW / 2, 8 + plotH);
      ctx.stroke();
    }

    ctx.font = "10px sans-serif";
    ctx.fillStyle = colors.text;
    ctx.fillText("Time step", plotW / 2 - 18, canvas.height - 6);
    ctx.fillText(String(steps), plotW - 20, canvas.height - 6);
  }, [
    canvasRef,
    raster,
    plotWidth,
    measuredHeight,
    highlightStep,
    emptyNote,
    highlightRows,
    sortByRate,
    colors,
  ]);
}
