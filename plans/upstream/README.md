# Upstream bug reports

Drafts of bug reports against this project's dependencies, written when the
bug was found and while the evidence was still in hand. They live here rather
than in `build/` so they survive a clean, and outside `documentation/` because
they are outbound reports rather than pages for this project's readers.

A draft stays here after filing, with the issue link added, so the evidence
that produced it does not have to be reconstructed later.

| Draft | Upstream | Status |
|---|---|---|
| [`tonic-hsd-float16-timestamps.md`](tonic-hsd-float16-timestamps.md) | [tonic](https://github.com/neuromorphs/tonic) | not filed |
| [`tonic-nmnist-dead-download-url.md`](tonic-nmnist-dead-download-url.md) | [tonic](https://github.com/neuromorphs/tonic) | not filed |

Both were found while adding an audio reference checkpoint; the float16 one is
the serious one — it silently destroys every timestamp in SHD and SSC, so a
network trains on a single time bin and still reports a plausible accuracy.
`spikeforge/events/hsd_reader.py` is this project's way around it, and
`EventTimestampError` refuses the corrupted stream if it is ever hit directly.
