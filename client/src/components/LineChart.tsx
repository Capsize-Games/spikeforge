import { useEffect, useRef } from "react";

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
}

export function LineChart({ series, title, width = 340, height = 150 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const pad = 28;
    const w = canvas.width - pad * 2;
    const h = canvas.height - pad;

    ctx.strokeStyle = "#30363d";
    ctx.strokeRect(pad, 8, w, h);

    series.forEach((s) => {
      if (s.values.length < 2) return;
      const max = s.max ?? Math.max(...s.values, 1e-6);
      ctx.beginPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      s.values.forEach((v, i) => {
        const x = pad + (i / (s.values.length - 1)) * w;
        const y = 8 + h - (Math.max(0, v) / max) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    ctx.font = "10px sans-serif";
    let legendX = pad;
    series.forEach((s) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(legendX, canvas.height - 12, 8, 8);
      ctx.fillStyle = "#8b949e";
      ctx.fillText(s.label, legendX + 12, canvas.height - 5);
      legendX += ctx.measureText(s.label).width + 34;
    });
  }, [series, width, height]);

  return (
    <div className="panel">
      {title && <div className="panel-title">{title}</div>}
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  );
}
