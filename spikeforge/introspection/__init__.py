"""Pure introspection: trajectories, spike codings, and surrogates.

Every function here is pure and returns plain Python values.
:func:`metrics.trajectory_metrics` gathers the trajectory metrics (firing
rate, ISI, sparsity, and histograms) into one dict,
:func:`encoding.encoding_report` reconstructs an image from a spike coding
with its coding-specific stats, and :mod:`surrogate` exposes the selectable
surrogate gradients and their derivative curves. Each result is
JSON-serialisable so later phases can stream it over the wire.
"""
