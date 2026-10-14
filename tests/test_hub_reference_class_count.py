"""The class count comes from the artifact, not the preset default."""

from pathlib import Path
from typing import Any

import torch


def _checkpoint(tmp_path: Path, **params: Any) -> str:
    """Write a checkpoint whose meta card declares its own class count."""
    from spikeforge.topology import registry

    _spec, module = registry.build_topology("fc_legacy", params)
    path = tmp_path / "probe.pt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "meta": {"num_classes": params["num_classes"]},
        },
        path,
    )
    return str(path)


def test_the_report_carries_the_declared_class_count(tmp_path: Path) -> None:
    """Inspection reads the count from the metadata card."""
    from spikeforge_hub.artifact_readers import inspect_torch

    report = inspect_torch(
        _checkpoint(tmp_path, num_classes=26, input_size=784)
    )
    assert report.num_classes == 26
    assert report.to_dict()["num_classes"] == 26


def test_a_bare_state_dict_declares_no_class_count(tmp_path: Path) -> None:
    """No metadata card means nothing is adopted, and nothing is invented."""
    from spikeforge.topology import registry
    from spikeforge_hub.artifact_readers import inspect_torch

    _spec, module = registry.build_topology("fc_legacy", {"num_classes": 26})
    path = tmp_path / "bare.pt"
    torch.save(module.state_dict(), path)
    assert inspect_torch(str(path)).num_classes is None


def test_a_non_ten_class_checkpoint_is_now_mappable(tmp_path: Path) -> None:
    """The readout size is the artifact's business, not the preset default.

    Before this, every checkpoint trained on a dataset without exactly ten
    classes was reported `incompatible` on a readout shape mismatch that said
    nothing about compatibility -- only that the comparison preset had been
    built with the wrong output size.
    """
    from spikeforge_hub.artifact_readers import inspect_torch
    from spikeforge_hub.compat import classify

    verdict = classify(
        inspect_torch(_checkpoint(tmp_path, num_classes=26, input_size=784))
    )
    assert verdict.mismatches == ()
    assert verdict.topology == "fc_legacy"


def test_the_ten_class_path_is_unchanged(tmp_path: Path) -> None:
    """Adopting the count must not disturb the entries already shipped."""
    from spikeforge_hub.artifact_readers import inspect_torch
    from spikeforge_hub.compat import classify

    verdict = classify(
        inspect_torch(_checkpoint(tmp_path, num_classes=10, input_size=784))
    )
    assert verdict.mismatches == ()


def test_only_the_class_count_is_adopted(tmp_path: Path) -> None:
    """Every other dimension stays something the comparison can still catch.

    A cochlea checkpoint declares 35 classes *and* 700 inputs. The class count
    is adopted; the input width is not, so it is still reported -- which is the
    point. A comparison that adopted both dimensions of the first layer could
    no longer disagree with any artifact about its shape.
    """
    from spikeforge_hub.artifact_readers import inspect_torch
    from spikeforge_hub.compat import classify

    verdict = classify(
        inspect_torch(_checkpoint(tmp_path, num_classes=35, input_size=700))
    )
    reasons = [dict(m) for m in verdict.mismatches]
    assert len(reasons) == 1
    assert reasons[0]["stage"] == "_fc1"
    assert reasons[0]["expected"] == "[128, 784]"
    assert reasons[0]["actual"] == "[128, 700]"
