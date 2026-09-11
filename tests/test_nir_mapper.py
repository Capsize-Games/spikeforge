"""Stage and neuron mapping onto NIR node parameters."""

import math
from typing import Any, Mapping

import pytest

from spikeforge.nir_bridge import api
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.mapper import map_stage
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage

pytest.importorskip("nir")


def _graph(kind: str, params: Mapping[str, Any]) -> Any:
    """Export a one-stage neuron chain and return its NIR graph."""
    return to_nir(chain([Stage("n", kind, params)]))


def _neuron_node(kind: str, params: Mapping[str, Any]) -> Any:
    """Export a one-stage neuron chain and return its named NIR node."""
    return _graph(kind, params).nodes["n"]


def test_leaky_renders_li_integrator_with_unit_gain() -> None:
    """A leaky stage becomes an LI plus threshold whose gain is exactly one."""
    beta = 0.5
    graph = _graph("leaky", {"beta": beta})
    integrator = graph.nodes["n__mem"]
    assert type(integrator).__name__ == "LI"
    assert float(integrator.tau) == pytest.approx(-1.0 / math.log(beta))
    assert float(integrator.r) == pytest.approx(1.0 / (1.0 - beta))
    assert type(graph.nodes["n"]).__name__ == "Threshold"
    assert float(graph.nodes["n"].threshold) == pytest.approx(1.0)


def test_subtract_and_zero_resets_render_differently() -> None:
    """Subtract feedback scales by -threshold; zero reset is a hard LIF."""
    subtract = _graph("leaky", {"threshold": 2.0, "reset": "subtract"})
    assert float(subtract.nodes["n__reset_scale"].scale) == pytest.approx(-2.0)
    assert float(subtract.nodes["n"].threshold) == pytest.approx(2.0)

    zero = _graph("leaky", {"threshold": 2.0, "reset": "zero"})
    assert type(zero.nodes["n"]).__name__ == "LIF"
    assert float(zero.nodes["n"].v_reset) == pytest.approx(0.0)


def test_lapicque_maps_to_li_and_threshold() -> None:
    """A Lapicque stage becomes an LI integrator plus a Threshold node."""
    graph = _graph("lapicque", {"beta": 0.9})
    assert type(graph.nodes["n__mem"]).__name__ == "LI"
    assert type(graph.nodes["n"]).__name__ == "Threshold"


def test_synaptic_maps_to_cubalif() -> None:
    """A synaptic stage becomes a CubaLIF carrying tau_syn and tau_mem."""
    alpha, beta = 0.8, 0.5
    node = _neuron_node("synaptic", {"alpha": alpha, "beta": beta})
    assert type(node).__name__ == "CubaLIF"
    assert float(node.tau_syn) == pytest.approx(-1.0 / math.log(alpha))
    assert float(node.tau_mem) == pytest.approx(-1.0 / math.log(beta))


def test_linear_without_bias_maps_to_linear_node() -> None:
    """A bias-free linear stage maps to the bias-free NIR Linear node."""
    spec = chain(
        [
            Stage(
                "fc",
                "linear",
                {"in_features": 3, "out_features": 2, "bias": False},
            )
        ]
    )
    assert type(to_nir(spec).nodes["fc"]).__name__ == "Linear"


def test_unknown_kind_raises_named_error() -> None:
    """An unmappable kind raises a typed error naming the kind."""
    with pytest.raises(UnsupportedStageError) as excinfo:
        map_stage(Stage("weird", "does_not_exist", {}))
    assert excinfo.value.kind == "does_not_exist"
    assert "does_not_exist" in str(excinfo.value)


def test_missing_primitive_raises_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage whose primitive is unavailable fails loudly, not silently."""
    monkeypatch.setattr(api, "node_class", lambda name: None)
    with pytest.raises(UnsupportedStageError) as excinfo:
        map_stage(Stage("lif", "leaky", {"beta": 0.5}))
    assert excinfo.value.kind == "leaky"
    assert "installed nir is missing" in str(excinfo.value)
