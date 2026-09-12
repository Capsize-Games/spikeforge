"""Relational Reversal Task falsification harness (issue #26).

Tests "The Consciousness Gradient" paper's own pre-committed,
falsifiable benchmark (Section 9.2) against the most parametrically
determined architecture available: a frozen, meta-trained predictor
whose only adaptation mechanism is conditioning on growing in-context
history, never a weight update. Deliberately separate from the
memory-system architecture roadmap in :mod:`spikeforge.memory`
(issues #22-#25): this package tests a claim, not a feature.
"""
