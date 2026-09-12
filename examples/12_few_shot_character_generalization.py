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
"""

from spikeforge.memory.few_shot_generalization_poc import (
    run_few_shot_generalization_poc,
)


def main() -> int:
    """Run the protocol once and print the cross-script accuracy."""
    result = run_few_shot_generalization_poc()
    print(
        f"one-shot 5-way KMNIST accuracy after training only on "
        f"EMNIST letters: {result.accuracy:.1%} "
        f"(chance = {result.chance:.1%}, {result.episodes} episodes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
