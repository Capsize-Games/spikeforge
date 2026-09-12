"""Few-shot generalization to a script never trained on (issue #24).

Trains a spiking embedder with episodic (prototypical-network) loss
on EMNIST letters (English), freezes it, then tests one-shot
discrimination among KMNIST (Japanese Kuzushiji) characters using
#22's unchanged ``OneShotAssociativeMemory`` as the readout.

Run from the repository root::

    venv/bin/python examples/12_few_shot_character_generalization.py

Unlike issue #22, this is a genuine research question, not an
engineering guarantee: the embedder must generalise its notion of
"same character, different instance" across an entirely different
script, something no amount of memory-side cleverness can fake.

Runs both a flat (``fc_small``) and a spatial (``conv_net``) embedder:
an ablation ruling out "the flat embedder throws away spatial
structure" as the explanation for a chance-level result, alongside
the shot-count and capacity ablations recorded in
plans/memory_system_research.md.
"""

from spikeforge.memory.few_shot_generalization_poc import (
    FewShotResult,
    run_few_shot_generalization_poc,
)


def _report(label: str, result: FewShotResult) -> None:
    """Print one topology's cross-script accuracy."""
    print(
        f"{label}: one-shot 5-way KMNIST accuracy after training only "
        f"on EMNIST letters: {result.accuracy:.1%} "
        f"(chance = {result.chance:.1%}, {result.episodes} episodes)"
    )


def main() -> int:
    """Run the protocol for both embedder topologies and print both."""
    _report("fc_small", run_few_shot_generalization_poc())
    _report(
        "conv_net",
        run_few_shot_generalization_poc(topology="conv_net", hidden=8),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
