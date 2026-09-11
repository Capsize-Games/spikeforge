"""Run the sequence experiments: exportable MLP versus attention.

``sequence_mlp`` is built only from NIR-mappable kinds, so it exports and
validates end to end. ``sequence_attn`` uses embedding/attention/layer-norm
kinds the installed ``nir`` cannot represent, so ``to_nir`` raises the typed
``UnsupportedStageError`` naming the first unexportable stage; it stays
runnable in the snnTorch simulator.

Run from the repository root::

    venv/bin/python examples/08_sequence_experiments.py

This enables sequence and attention *experimentation*, not production LLM
training.
"""

from spikeforge.cli import fixture
from spikeforge.nir_bridge import to_nir, validate
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.simulator.runner import run


def exportable() -> None:
    """Export and validate ``sequence_mlp`` end to end."""
    spec, module, frames = fixture.sequence_input("sequence_mlp", 4, 1, 0)
    to_nir(spec, module)
    report = validate(spec, module, frames)
    print("sequence_mlp frames:", tuple(frames.shape))
    print("sequence_mlp within_tolerance:", report["within_tolerance"])


def simulation_only() -> None:
    """Show ``sequence_attn`` export failing, yet still simulating."""
    spec, module, frames = fixture.sequence_input("sequence_attn", 4, 1, 0)
    try:
        to_nir(spec, module)
    except UnsupportedStageError as error:
        print(f"sequence_attn export refused: kind={error.kind}")
        print("  reason:", error)
    trajectory = run(module, frames)
    print(
        "sequence_attn simulated logits:",
        tuple(trajectory.logits.shape),
    )


def main() -> int:
    """Run both sequence experiments."""
    exportable()
    simulation_only()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
