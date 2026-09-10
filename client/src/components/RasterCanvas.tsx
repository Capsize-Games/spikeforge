import { useEffect, useRef } from "react";
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
}

export function RasterCanvas({
  raster,
  width = 600,
  height = 260,
  label,
  actions,
  highlightStep = null,
  emptyNote,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (!raster) {
      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (emptyNote) {
        ctx.fillStyle = "#8b949e";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(emptyNote, canvas.width / 2, canvas.height / 2);
        ctx.textAlign = "start";
      }
      return;
    }

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const steps = Math.max(1, raster.num_steps);
    const neurons = Math.max(1, raster.num_neurons);
    const padL = 46;
    const padB = 22;
    const plotW = canvas.width - padL - 12;
    const plotH = canvas.height - padB - 14;

    ctx.strokeStyle = "#30363d";
    ctx.strokeRect(padL, 8, plotW, plotH);

    ctx.fillStyle = "#c9d1d9";
    ctx.font = "10px sans-serif";
    ctx.fillText(String(neurons), 4, 16);

    ctx.fillStyle = "#e3b341";
    for (let i = 0; i < raster.time.length; i++) {
      const x = padL + (raster.time[i] / steps) * plotW;
      const y = 8 + plotH - (raster.neurons[i] / neurons) * plotH;
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
        const y = 8 + plotH - (raster.neurons[i] / neurons) * plotH;
        ctx.fillRect(x, y, 2.4, 2.4);
      }
      ctx.strokeStyle = "rgba(88, 166, 255, 0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + bandW / 2, 8);
      ctx.lineTo(x0 + bandW / 2, 8 + plotH);
      ctx.stroke();
    }

    ctx.fillStyle = "#8b949e";
    ctx.fillText("Time step", padL + plotW / 2 - 18, canvas.height - 6);
    ctx.fillText(String(steps), padL + plotW - 20, canvas.height - 6);
  }, [raster, width, height, highlightStep, emptyNote]);

  return (
    <div className="panel">
      {(label || actions) && (
        <div className="panel-title row-title">
          {label && <span>{label}</span>}
          {actions && <span className="panel-actions">{actions}</span>}
        </div>
      )}
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  );
}
