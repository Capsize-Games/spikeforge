/**
 * Semantic plot colours for the training charts, from the plot palette:
 * coral for the loss (error), emerald for held-out accuracy (positive), and
 * desaturated slate for the train-batch reference trace.
 */
export const CHART_COLORS = {
  loss: "#f85149",
  heldOut: "#3fb950",
  trainBatch: "#64748b",
} as const;
