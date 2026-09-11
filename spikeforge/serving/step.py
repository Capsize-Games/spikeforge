"""The shared per-step sweep, re-exported for the serving package.

Serving does not own a temporal loop of its own: it drives the same
:func:`~spikeforge.simulator.step_stages.step_stages` the closed loop uses, so
streaming and batch inference cannot diverge.
"""

from spikeforge.simulator.step_stages import step_stages

__all__ = ["step_stages"]
