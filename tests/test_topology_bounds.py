"""DoS bounds on client-controlled topology sizes.

Oversized ``topology_params`` are rejected by ``build_topology``
before any spec or module -- and therefore any tensor -- is built. A
passing rejection test is itself the proof: allocating
``hidden=10**9`` would OOM the runner instead of raising.
"""

from typing import Any, Dict

import pytest

from spikeforge.topology import registry, size_bounds


def test_registry_defaults_build_unchanged() -> None:
    """Every shipped topology still builds from its defaults."""
    for name in registry.topology_names():
        registry.build_topology(name)


@pytest.mark.parametrize(
    "name,params",
    [
        ("fc_legacy", {"hidden": 10 ** 9}),
        ("fc_legacy", {"hidden": 0}),
        ("fc_legacy", {"hidden": -4}),
        ("fc_legacy", {"hidden": True}),
        ("fc_legacy", {"num_classes": 10 ** 9}),
        ("fc_legacy", {"input_size": 10 ** 9}),
        ("fc_small", {"hidden": 10 ** 9}),
        ("conv_net", {"channels": 10 ** 9}),
        ("conv_net", {"channels": 0}),
        ("conv_net", {"in_channels": -1}),
        ("conv_net", {"input_size": (10 ** 5, 10 ** 5)}),
        ("conv_net", {"input_size": 0}),
        ("conv_net", {"input_size": (28, 0)}),
        ("recurrent_net", {"hidden": 10 ** 9}),
        ("sequence_mlp", {"hidden": 10 ** 9}),
        ("sequence_mlp", {"seq_length": 10 ** 9}),
        ("sequence_mlp", {"features": 0}),
        ("sequence_attn", {"vocab": 10 ** 9}),
        ("sequence_attn", {"embed_dim": 10 ** 9}),
        ("sequence_attn", {"num_heads": 0}),
        ("sequence_attn", {"num_heads": 10 ** 9}),
    ],
)
def test_absurd_topology_params_rejected(
    name: str, params: Dict[str, Any]
) -> None:
    """Absurd sizes fail validation instead of allocating layers."""
    with pytest.raises(ValueError):
        registry.build_topology(name, params)


def test_boundary_values_validate_without_building() -> None:
    """Each cap accepts its edge; one past the edge fails, unbuilt."""
    size_bounds.validate_params(
        "fc_small", {"hidden": size_bounds.MAX_HIDDEN}
    )
    with pytest.raises(ValueError):
        size_bounds.validate_params(
            "fc_small", {"hidden": size_bounds.MAX_HIDDEN + 1}
        )
    size_bounds.validate_params(
        "conv_net", {"channels": size_bounds.MAX_CHANNELS}
    )
    with pytest.raises(ValueError):
        size_bounds.validate_params("conv_net", {"channels": 0})


def test_rejection_precedes_module_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation runs before the module builder ever sees the params."""
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a rejected config must not allocate")

    monkeypatch.setattr(registry, "build_module", _explode)
    with pytest.raises(ValueError):
        registry.build_topology("fc_small", {"hidden": 10 ** 9})
