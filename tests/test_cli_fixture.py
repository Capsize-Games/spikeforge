"""The synthetic deployment fixtures are reproducible, weights included.

These assert a *property* -- that two calls agree -- rather than pinned
tensor values, so they keep their meaning across a torch upgrade that shifts
the RNG stream. A test pinning literal floats would fail on such an upgrade
without anything actually being wrong.

Each comparison sets the global torch RNG deliberately before every call,
because the defect was that a fixture inherited whatever ambient state it
happened to be called in. Two regimes are needed and they are opposites:

* reproducibility is only meaningful from *differing* ambient state, since
  two calls made back to back can agree purely by accident;
* seed sensitivity is only meaningful from *identical* ambient state, which
  leaves the seed as the sole variable -- against the unfixed code the seed
  had no effect on the weights whatsoever.

Each CLI invocation is its own process, so in real use the ambient state was
always the first of these.
"""

from typing import Any, List

import pytest
import torch

from spikeforge.cli import fixture

#: Presets whose fixture is the flat ``[T, B, F]`` volume.
DENSE = ("conv_net", "fc_legacy", "fc_small")
#: Presets routed through the sequence path, dense frames and tokens.
SEQUENCE = ("sequence_mlp", "sequence_attn")


def _fresh_ambient(step: int) -> None:
    """Leave the global RNG in a state unique to ``step``.

    Models two separate CLI invocations, whose ambient state has no reason
    to match.
    """
    torch.manual_seed(1000 + step)
    torch.rand(step + 1)


def _fixed_ambient() -> None:
    """Leave the global RNG in one state, whoever calls it.

    Holds everything but the seed constant, so a difference in the weights
    can only have come from the seed.
    """
    torch.manual_seed(20260915)
    torch.rand(17)


def _weights(module: Any) -> List[torch.Tensor]:
    """Return a module's parameters in a stable order."""
    return [value for _name, value in sorted(module.state_dict().items())]


def _same_weights(left: Any, right: Any) -> bool:
    """Return True when two modules hold identical parameter tensors."""
    first, second = _weights(left), _weights(right)
    if len(first) != len(second):
        return False
    return all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize("topology", DENSE + SEQUENCE)
def test_same_seed_reproduces_weights_and_spikes(topology: str) -> None:
    """Two calls at one seed agree on the module as well as the input.

    The seed used to be applied after the module was built, so the spikes
    reproduced while the weights did not. Every figure a deployment report
    derives from those weights -- per-layer ranges, quantization error,
    drift -- therefore changed on each invocation of a command documented
    as deterministic.
    """
    _fresh_ambient(0)
    _spec_a, module_a, spikes_a = fixture.synthetic_input(topology)
    _fresh_ambient(1)
    _spec_b, module_b, spikes_b = fixture.synthetic_input(topology)
    assert _same_weights(module_a, module_b)
    assert torch.equal(spikes_a, spikes_b)


@pytest.mark.parametrize("topology", DENSE + SEQUENCE)
def test_a_different_seed_gives_different_weights(topology: str) -> None:
    """The weights follow the seed rather than being frozen constants."""
    _fixed_ambient()
    _spec_a, module_a, _spikes_a = fixture.synthetic_input(topology, seed=0)
    _fixed_ambient()
    _spec_b, module_b, _spikes_b = fixture.synthetic_input(topology, seed=1)
    assert not _same_weights(module_a, module_b)


def test_the_input_volume_does_not_depend_on_the_topology() -> None:
    """One seed gives one input volume, whatever the preset consumed.

    The module is built from the same global RNG the volume is drawn from,
    so without re-seeding between the two a preset that gained a stage would
    silently change the spikes every other preset is compared on.
    """
    volumes = []
    for index, topology in enumerate(DENSE):
        _fresh_ambient(index)
        volumes.append(
            fixture.synthetic_input(topology)[2].reshape(
                fixture.STEPS, fixture.BATCH, -1
            )
        )
    for other in volumes[1:]:
        assert torch.equal(volumes[0], other)


def test_repeated_deployment_reports_agree() -> None:
    """The defect end to end: one report, twice, must be one report."""
    targets = pytest.importorskip("spikeforge_targets.cli.deploy_cli")
    _fresh_ambient(0)
    first = targets.deploy_report("conv_net", "lava_loihi2")
    _fresh_ambient(1)
    second = targets.deploy_report("conv_net", "lava_loihi2")
    assert first["quantization"]["layers"] == second["quantization"]["layers"]
    assert first["quantization"]["drift"] == second["quantization"]["drift"]
