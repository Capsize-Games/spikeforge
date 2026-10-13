/** Hover descriptions for each configurable setting. */

export const HELP: Record<string, string> = {
  event_modal:
    "This dataset is a neuromorphic event stream: each sample already " +
    "carries ON/OFF spikes in its own time bins, so rate/latency/delta/" +
    "random coding does not apply. The coding and step controls are " +
    "disabled and the recording's own raster and playback are shown.",
  event_steps:
    "Event samples define their own time bins from the recording; the " +
    "step count is shown for reference and is not re-encoded.",
  event_availability:
    "Events need the optional tonic package. Install the events extra " +
    "(pip install -e \".[events]\") to enable these datasets; until then " +
    "the picker marks them unavailable instead of failing silently.",
  event_training:
    "Event datasets train through the event stream path: each sample's " +
    "own ON/OFF bins feed the network directly, with no rate/latency " +
    "encoding. The topology must match the sensor geometry — conv_net " +
    "needs a square 28x28-like sensor, while a non-square sensor needs a " +
    "feature-input topology whose input_size equals the sensor area.",
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
  input_size:
    "Optional sensor geometry for spatial topologies, written HxW (e.g. " +
    "32x28). Blank keeps the default 28x28. A conv_net derives its feature " +
    "count from this shape, so it must match the sample you build for.",
  animate_hidden:
    "Stream the loaded model's hidden-layer activation frame each step " +
    "during playback, on top of the raster. Needs a trained or loaded " +
    "model; without one the panel names the reason instead of drawing one.",
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
  trajectory:
    "Per-step neuron state captured in educational mode: the mean membrane " +
    "potential U[t] and input current I[t] for each stage of the active " +
    "model. Production mode skips this capture, so the panel stays empty.",
  nir_graph:
    "The NIR summary of the active topology: one node per primitive " +
    "(affine, LIF, delay, pooling) and one edge per connection. Dashed " +
    "edges carry a time delay; skip and recurrent edges bow out of the " +
    "column grid.",
  nir_validation:
    "Drift report from the independent NIR interpreter. It re-runs the " +
    "exported graph from its node parameters alone and compares spikes and " +
    "membranes to the snnTorch model, naming the worst offending layer.",
  metrics:
    "Aggregate firing rate, sparsity, and inter-spike-interval statistics " +
    "for each stage of the active model, plus the firing-rate histogram. " +
    "Like the trajectory viewer it needs educational mode, which records " +
    "the per-step spikes these metrics are derived from.",
  encoding_report:
    "Decodes the currently configured sample back to an image using the " +
    "selected coding, and reports firing rate, sparsity, and how exact the " +
    "reconstruction is. It tracks the LEFT encoding controls, so change " +
    "them and refresh.",
  surrogate_curve:
    "Backward-pass derivative dS/dU of the selected surrogate gradient, " +
    "sampled over a fixed range. Surrogates shape training only; inference " +
    "never uses them, and the default uses snnTorch's built-in Fast Sigmoid.",
  benchmark:
    "Times forward (and backward) passes across topologies and modes on a " +
    "tiny fixture resolved from the current training settings. It is never " +
    "run automatically; press Run benchmark. Long measurements report as —.",
  benchmark_compiled:
    "Whether the benchmark run used torch.compile and, when it did not, " +
    "why. Compilation can speed up steady-state training but costs a one-" +
    "off warm-up, so the status text explains a fallback instead of " +
    "silently reporting a slower number.",
  benchmark_memory:
    "Peak memory for the run, attributed to CPU or GPU. It is the high-" +
    "water mark, so it tells you the batch size a device can actually " +
    "hold rather than what is allocated right now.",
  energy:
    "Accounts the topology's synaptic operations (SOP), dense multiply-" +
    "accumulates (MAC), accumulations (AC), and timesteps, then maps them " +
    "to the target's declared cost table. The numbers are estimates unless " +
    "a real device reports its own timing, and a target with no declared " +
    "table reports that honestly instead of guessing.",
  time_cursor:
    "The shared time cursor for the centre panels. Scrubbing or playing " +
    "moves every raster, the neuron-state trace, and the prediction bars " +
    "together, so a spike at t lines up across layers. Prev/next also " +
    "re-encodes the displayed sample, keeping the picture and its spikes " +
    "in sync.",
  activity:
    "Per-layer spike rasters for the input, hidden, and output stages. The " +
    "hidden layer is sorted by firing rate and silent neurons are reported " +
    "because a stage that never fires passes no gradient — usually the " +
    "first sign that a coding or beta is misconfigured.",
  tour:
    "Guided tours open a short lesson per snnTorch tutorial. Each step " +
    "highlights the controls or panel it is talking about, so the lesson " +
    "explains the dashboard in place instead of sending you to separate " +
    "documentation.",
  targets:
    "Deployment targets from the registry. Each row names its kind, the " +
    "pip extra that would install its SDK, and whether that SDK is " +
    "actually present. A target whose SDK is missing is shown as " +
    "unavailable rather than hidden, so the gap is visible up front.",
  deployment_report:
    "How the selected target would run the active topology: nodes it " +
    "supports, nodes it replaces with a substitute, and nodes it cannot " +
    "run at all — nothing is silently dropped. 'deployable' is true only " +
    "when the target is available and every node is supported. When a " +
    "model and sample are loaded the report adds the reference " +
    "interpreter's drift check; otherwise that section is omitted, never " +
    "faked.",
  hub:
    "Browse, download, inspect, and import curated models. The catalog " +
    "renders offline; only live Hugging Face search needs the hub extra, " +
    "and an entry that needs it is marked unavailable with the reason. " +
    "Import is a three-gate funnel — inspect, then compatibility, then " +
    "promote — so an artifact is mapped to a shipped preset or rejected " +
    "with the mismatched stages named, never loaded wrong.",
};

export const TRAIN_HELP: Record<string, string> = {
  dataset:
    "Which dataset to train on. Images are normalised to 28x28 grayscale; " +
    "event datasets train through the event stream path on their own " +
    "sensor, so a non-square sensor needs a feature-input topology. The " +
    "class count adjusts the output layer automatically.",
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
    "Compute device for training and inference. Auto (default) benchmarks " +
    "this exact configuration — encoding, transfer, forward and backward — " +
    "on CPU and GPU and picks the faster one; it falls back to CPU without " +
    "a GPU. For this small net CPU often wins because per-op launch " +
    "overhead dominates, while large hidden layers and batch sizes favour " +
    "the GPU. Watch 'ms/step' in Status to see the effect.",
  mode:
    "Execution mode for a run. Educational records per-step membrane, " +
    "current, and spike traces so the trajectory and metrics panels have " +
    "data; production skips that capture for a leaner run, so those panels " +
    "stay empty.",
  topology:
    "Network architecture from the topology registry. fc_legacy keeps the " +
    "original fully-connected LIF net; other presets add convolutional, " +
    "pooled, and recurrent stages.",
  neuron:
    "Neuron model used for every spiking stage, from the neuron registry. " +
    "leaky is the classic LIF; lapicque, alpha, synaptic, and recurrent " +
    "change how each membrane state evolves.",
  surrogate:
    "Surrogate gradient used to backprop through spikes. Default uses " +
    "snnTorch's built-in Fast Sigmoid; the other names are snnTorch's " +
    "surrogate factories and affect training only, never inference.",
  stage_neurons:
    "Override the neuron model for one named stage of the selected " +
    "topology, for example lif1 or lif2. Stage names come from the " +
    "topology; leave a stage out to use the Neuron setting above for it.",
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
