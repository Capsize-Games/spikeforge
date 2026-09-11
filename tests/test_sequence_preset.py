"""Demonstration sequence presets: validation, simulation, and honesty."""

import pytest
import torch

from spikeforge.cli import verify
from spikeforge.data import sequence_source
from spikeforge.nir_bridge import to_nir
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.simulator.runner import run
from spikeforge.topology import registry
from spikeforge_targets.rewrite import rewrite

pytest.importorskip("nir")


def test_registry_lists_both_sequence_presets() -> None:
    """Both sequence presets are registered and flagged as sequence."""
    assert registry.is_sequence_topology("sequence_mlp")
    assert registry.is_sequence_topology("sequence_attn")
    assert not registry.is_sequence_topology("fc_small")


def test_sequence_mlp_validates_end_to_end() -> None:
    """The NIR-mappable sequence preset stays within tolerance."""
    spec, module, spikes = verify.sample_input("sequence_mlp", "mnist")
    report = verify.validate_report(spec, module, spikes)
    assert report["within_tolerance"] is True


def test_sequence_mlp_graph_has_no_delay_node() -> None:
    """``reset='zero'`` renders a single LIF per stage, with no feedback."""
    spec, module = registry.build_topology("sequence_mlp")
    graph = to_nir(spec, module)
    kinds = [type(node).__name__ for node in graph.nodes.values()]
    assert "Delay" not in kinds
    assert kinds.count("LIF") == 2
    assert "Scale" not in kinds


def test_sequence_mlp_is_backend_runnable_for_norse() -> None:
    """The exported graph needs no substitution for the Norse target."""
    spec, module = registry.build_topology("sequence_mlp")
    spikes = sequence_source.random_frames(4, 2, 8, 8, 0)
    result = rewrite(to_nir(spec, module), "norse", spikes)
    assert result.report.unfixable == ()
    assert result.report.applied == ()
    assert result.report.ready()


def test_sequence_attn_simulates() -> None:
    """The attention demo runs its token stack in simulation."""
    _spec, module = registry.build_topology("sequence_attn")
    tokens = torch.randint(0, 32, (4, 2, 8))
    trajectory = run(module, tokens, track=True)
    assert tuple(trajectory.logits.shape) == (2, 8, 4)


def test_sequence_attn_export_names_embedding() -> None:
    """Export fails loudly, naming the first unexportable stage."""
    spec, module = registry.build_topology("sequence_attn")
    with pytest.raises(UnsupportedStageError) as excinfo:
        to_nir(spec, module)
    assert excinfo.value.kind == "embedding"


def test_sequence_attn_mixes_per_stage_neurons() -> None:
    """The attention preset accepts per-stage neuron overrides."""
    spec, module = registry.build_topology(
        "sequence_attn", {"neurons": {"out": "synaptic"}}
    )
    assert spec.stage("out").kind == "synaptic"
    tokens = torch.randint(0, 32, (3, 1, 8))
    assert run(module, tokens, track=True).steps == 3


def test_verify_validate_cli_accepts_sequence_mlp() -> None:
    """The CLI validate gate exits zero for the sequence MLP."""
    assert verify.main(["validate", "--topology", "sequence_mlp"]) == 0


def test_verify_export_cli_reports_unexportable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI export gate names the unexportable stage and exits non-zero."""
    code = verify.main(["export", "--topology", "sequence_attn"])
    out = capsys.readouterr().out
    assert code == 1
    assert "embedding" in out


def test_verify_validate_cli_reports_unexportable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI validate gate reports the unexportable stage cleanly."""
    code = verify.main(["validate", "--topology", "sequence_attn"])
    out = capsys.readouterr().out
    assert code == 1
    assert "embedding" in out
