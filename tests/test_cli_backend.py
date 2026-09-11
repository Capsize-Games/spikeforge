"""CLI payload shapes and exit codes for the rewrite and run commands."""

import json

import pytest

from spikeforge.cli import verify
from spikeforge_targets.cli import target_cli

pytest.importorskip("nir")


def test_cli_rewrite_prints_a_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``rewrite`` prints the rewrite report and exits success."""
    code = verify.main(
        ["rewrite", "--topology", "conv_net", "--target", "lava_loihi2"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "lava_loihi2"
    assert payload["applied"][0]["to"] == "SumPool2d"
    assert payload["drift"]["within_tolerance"] is True


def test_cli_run_reference_is_ok(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run`` on the reference backend exits success with a comparison."""
    code = verify.main(
        ["run", "--topology", "fc_small", "--target", "reference"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["compare"]["readout"]["max_abs"] <= 1e-6


def test_cli_run_unavailable_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run`` on an absent SDK exits non-zero with an honest status."""
    code = verify.main(
        ["run", "--topology", "fc_small", "--target", "norse"]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "unavailable"
    assert payload["notes"]


def test_cli_rewrite_is_also_on_the_targets_entry_point(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``spikeforge-targets`` entry point exposes ``rewrite``."""
    code = target_cli.main(
        ["rewrite", "--topology", "fc_small", "--target", "reference"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "reference"
    assert payload["ready"] is True


def test_cli_run_shapes_a_sequence_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run`` feeds a sequence topology a ``[T, B, L, D]`` fixture.

    Regression guard: the fixture must respect the sequence input layout,
    not the flat ``[T, B, 784]`` image volume, or the first dense stage
    rejects the sample before the backend can run.
    """
    code = verify.main(
        ["run", "--topology", "sequence_mlp", "--target", "reference"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["compare"]["readout"]["max_abs"] <= 1e-6


def test_cli_deploy_shapes_a_sequence_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``deploy`` classifies a sequence topology using the same fixture."""
    code = verify.main(
        ["deploy", "--topology", "sequence_mlp", "--target", "reference"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deployable"] is True
