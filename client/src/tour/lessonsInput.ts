import type { TourLesson } from "./types";

/** Input-side lessons: spike encoding and datasets. */
export const INPUT_LESSONS: TourLesson[] = [
  {
    id: "encoding",
    name: "Spike Encoding",
    summary: "Turn a static image into spikes and read it back.",
    steps: [
      {
        title: "Pick the dataset",
        body:
          "Encoding starts from a sample, and the dataset picker " +
          "chooses which image pool feeds both the viewer and " +
          "training. Keeping them on the same dataset is what makes " +
          "the picture you see the same data the network learns.",
        target: '[data-tour="dataset"]',
      },
      {
        title: "Step through samples",
        body:
          "Prev/next re-encode the newly selected sample, so the " +
          "raster and the encoding report always describe the same " +
          "image. Stepping the index is the quickest way to see how " +
          "a coding behaves across differently bright digits.",
        target: '[data-tour="sample-index"]',
      },
      {
        title: "Choose a coding",
        body:
          "Rate, latency, delta, and random each answer 'when does " +
          "a neuron fire?'. Rate encodes brightness as firing " +
          "frequency, latency as first-spike timing, delta fires on " +
          "large changes, and random is the noise baseline.",
        target: '[data-tour="coding"]',
      },
      {
        title: "Watch the input",
        body:
          "The centre row shows the raw sample, the current spike " +
          "frame, and the decoded image. Reading them together is " +
          "how you judge whether a coding preserves the information " +
          "the classifier needs.",
        target: '[data-tour="input-raster"]',
      },
      {
        title: "Read the reconstruction",
        body:
          "The encoding report decodes the sample back to an image " +
          "and reports firing rate, sparsity, and how exact the " +
          "round-trip is. A poor reconstruction means the code threw " +
          "away information before the network ever saw it.",
        target: '[data-tour="encoding-report"]',
        absentNote:
          "The encoding report is in the right-hand analysis " +
          "column; it appears once the server is connected and a " +
          "sample is configured.",
      },
    ],
  },
  {
    id: "datasets",
    name: "Neuromorphic Datasets",
    summary: "Datasets, sample navigation, and where event data lands.",
    steps: [
      {
        title: "The dataset picker",
        body:
          "Every dataset is normalised to 28x28 grayscale and its " +
          "class count resizes the output layer automatically, so " +
          "switching datasets cannot silently mismatch the network " +
          "and its labels.",
        target: '[data-tour="dataset"]',
      },
      {
        title: "Navigate samples",
        body:
          "Use the sample stepper to move through the dataset. Each " +
          "move re-encodes the new sample and re-runs the loaded " +
          "model, keeping the sample, its spikes, and the prediction " +
          "in sync.",
        target: '[data-tour="sample-index"]',
      },
      {
        title: "Where event data lands",
        body:
          "snnTorch's neuromorphic datasets stream asynchronous " +
          "events rather than pixels. This dashboard currently " +
          "ingests static images; event datasets such as DVS or " +
          "N-MNIST arrive in Phase 4, which is why no event picker " +
          "is shown yet.",
        target: '[data-tour="input-raster"]',
      },
    ],
  },
];
