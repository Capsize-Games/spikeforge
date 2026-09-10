import { useEffect, useRef } from "react";

import { useTheme } from "../theme";

interface Series {
  label: string;
  color: string;
  values: number[];
  /** Optional max for normalising (e.g. 1.0 for accuracy). */
  max?: number;
}

interface Props {
  series: Series[];
  title?: string;
  width?: number;
  height?: number;
  /** Draw a vertical marker at this sample index (e.g. the time cursor). */
  cursorIndex?: number | null;
  /** Render only the canvas, for nesting inside another panel. */
  bare?: boolean;
}

export function LineChart({
  series,
  title,
  width = 340,
  height = 150,
  cursorIndex = null,
  bare = false,
}: Props) {
  const { colors } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // No frame and no padding: the plot fills the canvas, leaving only a thin
    // strip at the bottom for the legend.
    const top = 2;
    const legendH = 16;
    const w = canvas.width;
    const h = canvas.height - top - legendH;

    series.forEach((s) => {
      if (s.values.length < 2) return;
      const max = s.max ?? Math.max(...s.values, 1e-6);
      ctx.beginPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      s.values.forEach((v, i) => {
        const x = (i / (s.values.length - 1)) * w;
        const y = top + h - (Math.max(0, v) / max) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Vertical marker tying the trace to the shared time cursor.
    const count = series[0]?.values.length ?? 0;
    if (cursorIndex !== null && count > 1) {
      const idx = Math.max(0, Math.min(cursorIndex, count - 1));
      const x = (idx / (count - 1)) * w;
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + h);
      ctx.stroke();
    }

    ctx.font = "10px sans-serif";
    let legendX = 2;
    series.forEach((s) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(legendX, canvas.height - 12, 8, 8);
      ctx.fillStyle = colors.text;
      ctx.fillText(s.label, legendX + 12, canvas.height - 5);
      legendX += ctx.measureText(s.label).width + 34;
    });
  }, [series, width, height, cursorIndex, colors]);

  if (bare) {
    return (
      <canvas
        ref={canvasRef}
        className="chart-canvas"
        width={width}
        height={height}
      />
    );
  }

  return (
    <div className="panel">
      {title && <div className="panel-title">{title}</div>}
      <canvas
        ref={canvasRef}
        className="chart-canvas"
        width={width}
        height={height}
      />
    </div>
  );
}
