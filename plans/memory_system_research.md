# Spikeforge — Memory-system research track (issues #22-#26)

> Research track, not a PC-0 production use case: these issues test what a
> genuinely spiking, one-shot, non-forgetting memory can and cannot do,
> rather than shipping a scoped product feature. Every claim about current
> code is grounded in a file/line reference; sections marked **design
> only** describe work that has not been built yet.

## 1. Why this track exists

Spikeforge's existing topologies (`fc_small`, `conv_net`, ...) are trained
once, end to end, against a fixed set of classes
([`presets.py`](spikeforge/topology/presets.py:1)). That is the right shape
for a static classifier and the wrong shape for a question this track
asks instead: **can a spiking system learn something new, on the spot,
from a single example, without a retraining cycle and without disturbing
what it already knew?** "No forgetting" is easy to fake with a classical
bolt-on (a nearest-neighbour cache, an embedding database); the point of
this track is to answer it with mechanisms that are actually spiking —
local synaptic plasticity and LIF dynamics, not a lookup table.

## 2. Issue #22 — held-out-digit POC (shipped)

**What it demonstrates.** `fc_small` is trained on MNIST digits 0-8 only,
then frozen (`net.requires_grad_(False)` in
[`frozen_classifier.py`](spikeforge/memory/frozen_classifier.py:34)). A
separate module,
[`OneShotAssociativeMemory`](spikeforge/memory/one_shot_associative_memory.py:18),
is taught digit 9 from a **single** example: one outer-product Hebbian
write ([`HebbianSynapse.write`](spikeforge/memory/hebbian_synapse.py:38))
into a dedicated LIF neuron's incoming weights — no gradient, no
optimizer step. Recall is ordinary `snn.Leaky` dynamics over that
synapse's current.

**Result (one real run, not cherry-picked; see
[`11_held_out_digit_memory.py`](examples/11_held_out_digit_memory.py:1)):**
retained accuracy on digits 0-8 held at **~91%** and recall accuracy on
the held-out 9 landed at **~23-26%** across a gain/threshold sweep (2x-16x
teach gain, 0.5/1.0 threshold) — well above the 10% ten-class chance
floor, short of reliable recognition. The retained number staying flat
across every sweep point is the structural guarantee working exactly as
designed: nothing in `teach()` can touch the frozen base net's weights.
The recall number is an honest research finding, not a tuned headline —
plausible explanation: the frozen net's hidden layer was never
incentivized to represent a class it never saw, so one exemplar's spike
pattern doesn't reliably generalize to other instances of that digit.
Follow-up ideas (not yet attempted): teaching from more than one
exemplar, or letting the base net's hidden layer train with a
class-agnostic (metric-learning) objective — the latter is exactly what
issue #24 explores, deliberately as a separate, harder claim.

**Reused by:** #23 and #24 both reuse
[`OneShotAssociativeMemory`](spikeforge/memory/one_shot_associative_memory.py:18)
unchanged as their memory readout.

## 3. Issue #23 — camera-fed room memory pipeline (design only)

Extension of #22 to a room-scale, always-on setting: a system that
watches a live feed, notices when something changed, identifies what
changed, and remembers whether it has seen *that specific instance*
before. Three stages, deliberately not one monolithic model:

```mermaid
flowchart LR
    A[Event camera feed\nDVS x,y,polarity,t] --> B[Watcher: SNN, always on]
    B -- "novelty spike" --> C[Labeler: conventional, runs on demand]
    C -- "label + crop" --> D[Memory: OneShotAssociativeMemory]
    D -- "seen before? / new instance" --> E[Per-instance recall result]
```

### 3.1 Watcher (stage 1)

Application #3 from the original scoping issue (#22's issue body): an
always-on module that spikes on motion/change rather than running full
perception continuously. **Reuse, do not rebuild:** this is exactly
UC-3's reference architecture —
[`use_case_event_camera_vision.md`](plans/use_case_event_camera_vision.md:1)
— event-camera ingestion through
[`spikeforge/events/`](spikeforge/events/event_bridge.py:1) into a
compact spiking net, already scoped down to `dvs128_gesture`-style
topologies. The watcher's "class" here is binary (`novelty` /
`no-novelty`) rather than gesture identity, which only changes the head,
not the ingestion or runtime story UC-3 already designed.

### 3.2 Labeler (stage 2)

A fixed, pretrained, non-spiking object-detection/scene-understanding
model, run only when the watcher flags novelty — mirrors the
face-embedding backbone pattern already discussed for application #1 in
#22's scoping issue. Not built from scratch; not part of this repo's
SNN-native surface, by design (per the project's SNN-native-for-the-
learning-mechanism stance — the labeler is perception, not memory).

### 3.3 Memory (stage 3)

The **unchanged** #22 module,
[`OneShotAssociativeMemory`](spikeforge/memory/one_shot_associative_memory.py:18).
The open integration question is the input contract: #22 feeds it
`fc_small`'s hidden-layer spikes
([`hidden_trace`](spikeforge/memory/lockstep_runner.py:15)); stage 3 here
needs the labeler's crop turned into a spike pattern of the same shape
(`[T, H]`) before `teach()`/`step()` can be called unchanged. The
simplest bridge — rate-code a fixed-size embedding of the crop through
the existing [`SpikeEncoder`](spikeforge/encoding/spike_encoder.py:8) —
needs no new code, only a decision on which embedding to rate-code.

### 3.4 Open questions (unresolved, tracked here rather than guessed at)

- Real DVS hardware vs. recorded/converted RGB for the watcher — depends
  on available hardware, same caveat UC-3 already carries.
- Which off-the-shelf labeler, and what embedding size becomes the
  memory module's `in_features`.
- Novelty judgment: motion-only vs. "doesn't match anything already
  tracked" (the latter would call the memory module itself during the
  watcher stage, not just after the labeler — an open design fork, not
  decided here).

### 3.5 Explicitly out of scope

Internet scraping for images of similar items — considered and dropped
in the original scoping (ToS/legal exposure, doesn't help the core claim
of recognizing *the same instance again*). Revisit only if the goal
shifts toward describing objects for a person rather than the system
remembering them itself.

**Status:** design only. Implementation is queued behind #24/#25 in the
issue tracker; this section is the durable record of the pipeline shape
so implementation does not re-derive it from scratch.

## 4. Issues #24 and #26

Tracked in their own sections once landed — #24 changes the base
network's training objective (episodic/metric learning) rather than
reusing `fc_small` as-is, and #26 is a falsification test of an unrelated
philosophical claim, not a memory-system feature. See the issues
themselves for current status.
