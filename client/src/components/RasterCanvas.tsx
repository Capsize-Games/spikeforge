import { useEffect, useRef } from "react";

import type { RasterPayload } from "../types";

interface Props {
  raster: RasterPayload | null;
  width?: number;
  height?: number;
  label?: string;
}

export function RasterCanvas({
  raster,
  width = 600,
  height = 260,
  label,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !raster) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

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

    ctx.fillStyle = "#8b949e";
    ctx.fillText("Time step", padL + plotW / 2 - 18, canvas.height - 6);
    ctx.fillText(String(steps), padL + plotW - 20, canvas.height - 6);
  }, [raster, width, height]);

  return (
    <div className="panel">
      {label && <div className="panel-title">{label}</div>}
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  );
}
