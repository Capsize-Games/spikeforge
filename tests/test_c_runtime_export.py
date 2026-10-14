"""C-runtime validation of real topology exports."""

from typing import Any, List, Tuple

import pytest

from spikeforge.nir_bridge.exporter import graph_summary, to_nir
from spikeforge.topology import presets, registry
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec, chain
from spikeforge.topology.stage import Stage
from spikeforge_targets.c_runtime import SUPPORTED
from spikeforge_targets.c_runtime_validate import validate_for_c_runtime

pytest.importorskip("nir")


def _pairs(target: Any) -> List[Tuple[str, str]]:
    """Return the ``(name, kind)`` violations for ``target``."""
    report = validate_for_c_runtime(target)
    return [(item.name, item.kind) for item in report.unsupported]


def test_built_conv_net_validates_clean() -> None:
    """The committed conv_net shape passes every C-loader assumption."""
    spec, module = registry.build_topology("conv_net")
    assert validate_for_c_runtime(to_nir(spec, module)).passed() is True


def test_structural_conv_net_validates_clean() -> None:
    """Placeholder weights still satisfy the C structural contract."""
    spec = presets.conv_net()
    assert validate_for_c_runtime(to_nir(spec)).passed() is True


def test_conv_net_stays_inside_the_declared_profile() -> None:
    """The conformance gate fails if export emits an undeclared kind."""
    spec, module = registry.build_topology("conv_net")
    summary = graph_summary(to_nir(spec, module))
    kinds = {node["kind"] for node in summary["nodes"]}
    assert kinds <= set(SUPPORTED), kinds - set(SUPPORTED)
    assert validate_for_c_runtime(summary).passed() is True


def test_validator_accepts_specs_and_graphs() -> None:
    """Specs export structurally; graphs validate as exported."""
    spec = presets.conv_net()
    assert validate_for_c_runtime(spec).passed() is True
    assert validate_for_c_runtime(to_nir(spec)).passed() is True


def test_zero_reset_lif_reports_the_neuron_by_name() -> None:
    """A zero-reset leaky stage exports LIF, which the C loader rejects."""
    spec = presets.fc_small(
        hidden=4,
        input_size=3,
        num_classes=2,
        stage_params={"lif1": {"reset": "zero"}},
    )
    assert ("lif1", "LIF") in _pairs(to_nir(spec, build_module(spec)))


def test_synaptic_stage_reports_cubalif_by_name() -> None:
    """Every synaptic stage maps to CubaLIF, outside the C profile."""
    spec = presets.fc_small(
        hidden=4,
        input_size=3,
        num_classes=2,
        neurons={"lif1": "synaptic"},
    )
    assert ("lif1", "CubaLIF") in _pairs(to_nir(spec, build_module(spec)))


def test_recurrent_stage_reports_lif_by_name() -> None:
    """A recurrent stage exports LIF plus its feedback delay."""
    spec = presets.fc_small(
        hidden=4,
        input_size=3,
        num_classes=2,
        neurons={"lif1": "recurrent"},
        stage_params={"lif1": {"linear_features": 4}},
    )
    assert ("lif1", "LIF") in _pairs(to_nir(spec, build_module(spec)))


def _linear_chain(bias: bool) -> TopologySpec:
    """Return a linear-plus-neuron chain with or without a bias."""
    return chain(
        [
            Stage(
                "fc1",
                "linear",
                {"in_features": 3, "out_features": 3, "bias": bias},
            ),
            Stage("lif1", "leaky", {"beta": 0.9}),
        ]
    )


def test_bias_free_linear_reports_linear_by_name() -> None:
    """Bias-free linears export as Linear, not the C-supported Affine."""
    spec = _linear_chain(bias=False)
    assert ("fc1", "Linear") in _pairs(to_nir(spec, build_module(spec)))


def test_biased_linear_validates_clean() -> None:
    """A biased linear exports Affine, inside the C profile."""
    spec = _linear_chain(bias=True)
    assert validate_for_c_runtime(to_nir(spec, build_module(spec))).passed()


def test_conv1d_reports_conv1d_by_name() -> None:
    """A conv1d stage exports Conv1d, outside the C profile."""
    spec = chain(
        [
            Stage(
                "conv",
                "conv1d",
                {
                    "in_channels": 2,
                    "out_channels": 2,
                    "kernel_size": 3,
                    "padding": 1,
                },
            ),
        ]
    )
    assert ("conv", "Conv1d") in _pairs(to_nir(spec, build_module(spec)))
