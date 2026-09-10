import { useEffect, useRef } from "react";

interface Props {
  data: number[][] | null;
  /** Heatmap color ramp name. */
  palette?: "binary" | "plasma";
  width?: number;
  height?: number;
  label?: string;
}

/** Color helpers for the two tutorial-style palettes. */
function color(value: number, palette: "binary" | "plasma"): string {
  const t = Math.max(0, Math.min(1, value));
  if (palette === "binary") {
    const v = Math.round(255 * t);
    return `rgb(${v},${v},${v})`;
  }
  const r = Math.round(255 * (1 - t * 0.2));
  const g = Math.round(255 * (0.2 + 0.7 * t));
  const b = Math.round(255 * (0.9 - 0.7 * t));
  return `rgb(${r},${g},${b})`;
}

export function HeatmapCanvas({
  data,
  palette = "binary",
  width = 200,
  height = 200,
  label,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = data.length;
    const cols = rows > 0 ? data[0].length : 0;
    if (rows === 0 || cols === 0) return;

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const cw = canvas.width / cols;
    const ch = canvas.height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = color(data[r][c], palette);
        ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
  }, [data, palette, width, height]);

  return (
    <div className="panel">
      {label && <div className="panel-title">{label}</div>}
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  );
}
