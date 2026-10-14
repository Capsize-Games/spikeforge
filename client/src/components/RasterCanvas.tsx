import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { max1, rowOrder, usePaintRaster } from "./rasterPaint";
import { useTheme } from "../theme";
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
  /**
   * Render without the outer panel chrome, for stacking several rasters
   * inside one shared panel (label + summary become one compact header row).
   */
  bare?: boolean;
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
  bare = false,
}: Props) {
  const { colors } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [plotWidth, setPlotWidth] = useState(width);
  // Seeded from the `height` prop so the first paint isn't zero-height;
  // once mounted, the wrap's own flex-allocated size takes over (see
  // .raster-row/.raster-plot/.raster-body in viewer.css) so the chart fills
  // whatever room the tab actually has instead of a guessed pixel constant.
  const [measuredHeight, setMeasuredHeight] = useState(height);

  // Track the panel's real size so canvas pixels map 1:1 to display pixels
  // on both axes — width keeps the time axis aligned with the Time cursor
  // slider, height lets the chart grow/shrink with the available viewport.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect;
      const cw = Math.floor(rect.width);
      const ch = Math.floor(rect.height);
      if (cw > 0) setPlotWidth(cw);
      if (ch > 0) setMeasuredHeight(ch);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  usePaintRaster({
    canvasRef,
    raster,
    plotWidth,
    measuredHeight,
    highlightStep,
    emptyNote,
    highlightRows,
    sortByRate,
    colors,
  });

  // Rail labels mirror the canvas row geometry so they line up with dots.
  const rowTop = 8;
  const plotAreaHeight = Math.max(1, measuredHeight - 22 - 14);
  const rowCount = raster ? max1(raster.num_neurons) : 0;
  const railLabels =
    raster && yLabels && yLabels.length === rowCount && rowCount <= 16
      ? rowOrder(raster, sortByRate).map((neuron, pos) => ({
          key: neuron,
          text: yLabels[neuron] ?? "",
          top: rowTop + plotAreaHeight - (pos / rowCount) * plotAreaHeight - 6,
        }))
      : [];

  // The rail sits outside the body (absolutely, to its left) so it never
  // narrows or shifts the canvas — every layer's plot keeps the same width.
  const body = (
    <div className="raster-plot">
      {railLabels.length > 0 && (
        <div className="raster-rail" style={{ height: measuredHeight }}>
          {railLabels.map(({ key, text, top }) => (
            <span key={key} className="rail-label" style={{ top }}>
              {text}
            </span>
          ))}
        </div>
      )}
      <div className="raster-body">
        <div className="raster-wrap" ref={wrapRef}>
          <canvas ref={canvasRef} width={plotWidth} height={measuredHeight} />
        </div>
      </div>
    </div>
  );

  // Bare mode shares one panel header with sibling rasters, so the label and
  // summary collapse into a single thin header line above the canvas.
  if (bare) {
    return (
      <div className="raster-row">
        {(label || summary || actions) && (
          <div className="raster-row-head">
            {label && <span className="raster-row-label">{label}</span>}
            {summary && <span className="raster-row-summary">{summary}</span>}
            {actions && <span className="panel-actions">{actions}</span>}
          </div>
        )}
        {body}
      </div>
    );
  }

  return (
    <div className="panel">
      {(label || actions) && (
        <div className="panel-title row-title">
          {label && <span>{label}</span>}
          {actions && <span className="panel-actions">{actions}</span>}
        </div>
      )}
      {body}
      {summary && <div className="raster-summary">{summary}</div>}
    </div>
  );
}
