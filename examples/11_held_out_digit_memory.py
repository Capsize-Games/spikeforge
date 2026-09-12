"""One-shot, no-forgetting memory: hold out a digit, teach it live.

Trains ``fc_small`` on MNIST digits 0-8, freezes it, then teaches
digit 9 to an SNN-native :class:`OneShotAssociativeMemory
<spikeforge.memory.one_shot_associative_memory.\
OneShotAssociativeMemory>` from a single example -- one local
Hebbian weight write, no backprop, no retraining of the base net.

Run from the repository root::

    venv/bin/python examples/11_held_out_digit_memory.py

This is issue #22's proof of concept: "no forgetting" is a structural
guarantee (the base classifier is frozen with ``requires_grad_(False)``
before teaching ever happens), so the interesting number is recall
accuracy on the digit it never saw during training.
"""

from spikeforge.memory.held_out_digit_poc import run_held_out_digit_poc


def main() -> int:
    """Run the protocol once and print both accuracy numbers."""
    result = run_held_out_digit_poc()
    print(f"retained accuracy (digits 0-8): {result.retained_accuracy:.1f}%")
    print(f"recall accuracy   (digit 9):    {result.recall_accuracy:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
