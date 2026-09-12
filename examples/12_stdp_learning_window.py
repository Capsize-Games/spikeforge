"""Trace the classic STDP learning-window curve from real spike pairs.

Sweeps isolated pre/post spike pairs at increasing separations through
:class:`~spikeforge.memory.stdp_synapse.STDPSynapse` and prints the
weight change each pairing produces. A causal pair (pre before post)
must potentiate, an anti-causal pair (post before pre) must depress,
and the effect must shrink with distance -- the signature shape of
spike-timing-dependent plasticity, not asserted but measured.

Run from the repository root::

    venv/bin/python examples/12_stdp_learning_window.py

This is the on-chip/local-learning line from the toolkit status table:
:class:`HebbianSynapse <spikeforge.memory.hebbian_synapse.HebbianSynapse>`
learns from a rate code (summed spike counts); this synapse instead
learns from spike order.
"""

from spikeforge.memory.stdp_window_poc import run_stdp_window_poc


def main() -> int:
    """Run the sweep once and print each offset's weight change."""
    print(f"{'delta_t':>8}  {'delta_w':>10}  timing")
    for point in run_stdp_window_poc():
        timing = "pre before post" if point.delta_t > 0 else "post before pre"
        print(f"{point.delta_t:>8}  {point.delta_w:>+10.4f}  {timing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
