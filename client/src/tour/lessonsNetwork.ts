import type { TourLesson } from "./types";

/** Network-side lessons: neurons, training, topologies, and NIR. */
export const NETWORK_LESSONS: TourLesson[] = [
  {
    id: "snn",
    name: "Spiking Neural Networks",
    summary: "Neuron models, educational mode, and the state viewer.",
    steps: [
      {
        title: "Choose a neuron model",
        body:
          "Every spiking stage uses the selected neuron from the " +
          "registry. leaky is the classic LIF; lapicque, alpha, " +
          "synaptic, and recurrent change how membrane state " +
          "evolves, which changes what the traces below should show.",
        target: '[data-tour="neuron"]',
      },
      {
        title: "Switch to educational mode",
        body:
          "Educational mode records per-step membrane, current, and " +
          "spike traces. Production skips that capture to run " +
          "leaner, so the trajectory viewer only has data to draw in " +
          "educational mode.",
        target: '[data-tour="mode-toggle"]',
      },
      {
        title: "Open the state viewer",
        body:
          "The neuron-state panel plots the mean membrane potential " +
          "U[t] and input current I[t] for one stage at a time. It " +
          "is the direct window into the dynamics behind a spike.",
        target: '[data-tour="trajectory"]',
      },
      {
        title: "Read membrane and current",
        body:
          "Pick a stage, then follow U[t] against I[t]: the " +
          "membrane charges while current flows and drops sharply " +
          "when a spike resets it. That reset is the event the " +
          "raster across the column is drawing.",
        target: '[data-tour="trajectory-chart"]',
        absentNote:
          "No trajectory is captured yet. In educational mode press " +
          "the ↻ on the neuron-state panel to record the active " +
          "model.",
      },
    ],
  },
  {
    id: "training",
    name: "Training SNNs",
    summary: "Surrogate gradients, the trainer, and live progress.",
    steps: [
      {
        title: "Pick the surrogate",
        body:
          "Spikes have no useful derivative, so training substitutes " +
          "the selected surrogate gradient. It shapes the backward " +
          "pass only; inference runs the true spiking dynamics.",
        target: '[data-tour="surrogate"]',
      },
      {
        title: "Inspect the surrogate curve",
        body:
          "The surrogate panel plots dS/dU over a fixed range. A " +
          "sharp peak gives crisp but sparse learning; a wide curve " +
          "keeps more neurons in the gradient, at the cost of " +
          "blurring the spike.",
        target: '[data-tour="surrogate-curve"]',
        absentNote:
          "Pick a named surrogate, then press ↻ on the surrogate " +
          "panel to sample its derivative; the built-in default has " +
          "no curve to draw.",
      },
      {
        title: "Start the fit",
        body:
          "Train builds a fresh network from the current topology, " +
          "neuron, and surrogate settings, then streams progress " +
          "back. Stop returns the trainer to idle without discarding " +
          "earlier history.",
        target: '[data-tour="train-controls"]',
      },
      {
        title: "Watch loss and accuracy",
        body:
          "The live readout shows step, epoch, loss, held-out " +
          "accuracy, and train-batch accuracy. Held-out tracking is " +
          "what tells you whether the network is generalising or " +
          "just memorising the subset.",
        target: '[data-tour="training-live"]',
        absentNote:
          "No run is active yet, so there is no live loss or " +
          "accuracy. Press Train model and this readout fills in as " +
          "steps complete.",
      },
    ],
  },
  {
    id: "cnn",
    name: "Spiking CNNs",
    summary: "Convolutional topologies and the export graph view.",
    steps: [
      {
        title: "Pick a convolutional topology",
        body:
          "A convolutional topology trades the fully-connected " +
          "hidden layer for conv and pooling stages, which share " +
          "weights across the image and preserve spatial structure. " +
          "Select one here to make it the active model for a run.",
        target: '[data-tour="topology"]',
      },
      {
        title: "Open the graph viewer",
        body:
          "The topology graph draws the active spec: one node per " +
          "primitive and one edge per connection. It is generated " +
          "from the same TopologySpec used to build the module, so " +
          "the picture cannot drift from the network it describes.",
        target: '[data-tour="nir-graph"]',
      },
      {
        title: "Read the layers",
        body:
          "Look for the conv and pooling nodes between input and " +
          "output. Press ↻ to export the graph for whatever topology " +
          "is currently configured.",
        target: '[data-tour="nir-graph-view"]',
        absentNote:
          "No graph is exported yet. Press ↻ on the topology graph " +
          "panel to summarise the active topology.",
      },
    ],
  },
  {
    id: "recurrent",
    name: "Recurrent SNNs",
    summary: "Recurrent topologies and delayed edges in the graph.",
    steps: [
      {
        title: "Select a recurrent topology",
        body:
          "A recurrent topology feeds a stage back into itself, " +
          "giving the network memory across time steps. Pick it " +
          "here, then export the graph to see how the loop is drawn.",
        target: '[data-tour="topology"]',
      },
      {
        title: "Find the recurrent edges",
        body:
          "In the graph, a recurrent edge bows out of the column " +
          "grid, and any dashed edge carries a time delay. Those are " +
          "the connections that make the model's output depend on " +
          "its own past.",
        target: '[data-tour="nir-graph"]',
      },
    ],
  },
  {
    id: "nir",
    name: "NIR",
    summary: "Export, validate against the reference, and benchmark.",
    steps: [
      {
        title: "Export the graph",
        body:
          "The NIR export flattens the active model into a graph of " +
          "primitives. Because the same spec renders both the " +
          "snnTorch module and the graph, an export is a faithful " +
          "view of the model rather than a separate description.",
        target: '[data-tour="nir-graph"]',
      },
      {
        title: "Validate the graph",
        body:
          "The drift panel re-runs the exported graph from its node " +
          "parameters alone, with no snnTorch involved, and compares " +
          "spikes and membranes to the model. It names the worst " +
          "offending layer, so a failure tells you where to look.",
        target: '[data-tour="nir-validation"]',
        absentNote:
          "No validation has run yet. Press ↻ on the drift panel to " +
          "compare the exported graph against the reference " +
          "interpreter.",
      },
      {
        title: "Benchmark the machine",
        body:
          "The benchmark times forward and backward passes across " +
          "topologies and modes on a tiny fixture aimed at your " +
          "settings. It never runs on its own; press Run benchmark, " +
          "then compare ms/step and steps/s. The compiled and status " +
          "columns record whether torch.compile engaged and why it " +
          "may have fallen back.",
        target: '[data-tour="benchmark"]',
      },
    ],
  },
];
