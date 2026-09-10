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
};
