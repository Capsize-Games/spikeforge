/** Hover descriptions for each configurable setting. */

export const HELP: Record<string, string> = {
  coding:
    "Spike encoding scheme. Rate uses firing frequency, latency uses " +
    "first-spike timing, delta fires on large changes, random generates " +
    "spikes from scratch.",
  num_steps:
    "Number of time steps in the simulation window. More steps give finer " +
    "temporal detail and longer animations.",
  subset:
    "Reduce the dataset by this factor. subset=10 keeps 1/10 of the data " +
    "(6000 MNIST samples) for faster encoding.",
  batch_size:
    "Number of samples processed together per batch. Larger batches are " +
    "faster but use more memory.",
  interval_ms:
    "Delay between animation frames in milliseconds. Higher values play " +
    "the spike animation back more slowly.",
  gain:
    "Scales spiking probability for rate coding. Lower gain produces " +
    "sparser spikes and a dimmer reconstruction.",
  vector_value:
    "Constant value used to build the demo raw vector before Bernoulli " +
    "encoding (the tutorial's 0.5 example).",
  tau:
    "RC time constant for latency coding. Higher tau slows the membrane " +
    "charging so spikes fire later.",
  threshold:
    "Membrane firing threshold for latency coding. Inputs below it are " +
    "clipped; lower values let dimmer pixels fire.",
  linear:
    "Use a linear latency curve instead of the logarithmic RC model. " +
    "Spreads firing times more evenly.",
  normalize:
    "Scale spike times to span the full num_steps range so no time steps " +
    "go unused.",
  clip:
    "Drop spikes that fall below the threshold (the dark background) " +
    "instead of piling them at the final time step.",
  delta_threshold:
    "Change in the time series required to emit a delta spike. Higher " +
    "thresholds fire less often.",
  random_scale:
    "Upper bound on the random spiking probability before conversion. " +
    "Higher values give denser random spike trains.",
  dataset:
    "Dataset the encoder reads the displayed sample from. Training uses " +
    "the same dataset so the network sees the encodings it is shown. " +
    "Changing it resets to sample index 0.",
  sample_index:
    "Which sample to display and encode. Prev/next step through the " +
    "dataset; the model is re-run on the new sample when one is loaded, " +
    "so the sample, its spikes, and the prediction stay in sync.",
  inference:
    "Run the loaded model on the currently encoded sample and report its " +
    "prediction, confidence, and the output-layer spike activity for each " +
    "class. Needs a trained or loaded model.",
};

export const TRAIN_HELP: Record<string, string> = {
  dataset:
    "Which dataset to train on. All are normalised to 28x28 grayscale; " +
    "the class count adjusts the output layer automatically.",
  hidden:
    "Number of neurons in the hidden LIF layer between the input and " +
    "output layers.",
  beta:
    "Membrane potential decay rate of each LIF neuron. Higher beta means " +
    "the neuron remembers its state longer.",
  lr: "Adam learning rate. Larger values learn faster but can be unstable.",
  epochs: "How many full passes over the training subset to run.",
  num_steps:
    "Time steps simulated per sample. The network accumulates output " +
    "spikes across these steps.",
  subset:
    "Dataset reduction factor for training. Higher subset = less data = " +
    "faster runs.",
  batch_size: "Number of samples per training batch.",
  device:
    "Compute device for training and inference. GPU defaults on when " +
    "available; if not, it silently falls back to CPU. Changing device " +
    "starts a fresh engine on your next Train.",
  resources:
    "Live CPU RAM and GPU VRAM for the machine running the server. CPU RAM " +
    "covers the whole host; VRAM is the GPU's memory. 'active' shows the " +
    "device the model is actually using.",
  input_mode:
    "How the checkpoints feeds samples into the network: a spike coding " +
    "(rate/latency/delta) or the legacy raw-pixel mode. Inference must " +
    "match the checkpoint's input mode.",
  compatibility:
    "A loaded checkpoint stores the dataset, coding, and step count it was " +
    "trained with. If the current encoding controls disagree, the banner " +
    "warns you: legacy raw checkpoints ignore the encoding controls, and a " +
    "dataset mismatch makes the output class count unsafe to compare, so " +
    "inference is disabled.",
};
