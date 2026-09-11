"""Capability classification of shipped presets against targets."""

import json

import pytest

from spikeforge.topology import presets
from spikeforge_targets import (
    EMITTED_PRIMITIVES,
    capability_matrix,
    registry,
)
from spikeforge_targets.node_view import node_kinds
from spikeforge_targets.target_spec import TargetSpec

pytest.importorskip("nir")

_REFERENCE = registry.get_target("reference")

_PRESETS = {
    "fc_legacy": presets.fc_legacy(
        hidden=8, beta=0.5, num_classes=3, input_size=5
    ),
    "conv_net": presets.conv_net(
        in_channels=1, channels=2, num_classes=3, input_size=8
    ),
    "recurrent_net": presets.recurrent_net(
        hidden=6, beta=0.9, num_classes=3, input_size=4
    ),
}

#: A target that keeps every emitted primitive except pooling.
_NARROW = TargetSpec(
    name="narrow",
    kind="simulator",
    description="Synthetic target with no pooling primitive.",
    extra=None,
    supported=frozenset(
        {
            "Input",
            "Output",
            "Conv2d",
            "Flatten",
            "Affine",
            "Linear",
            "LI",
            "Threshold",
            "Scale",
            "Delay",
        }
    ),
)

#: A target that sums where the source averages, declaring a substitute.
_SUBSTITUTING = TargetSpec(
    name="substituting",
    kind="simulator",
    description="Synthetic target that sums instead of averaging.",
    extra=None,
    supported=frozenset(EMITTED_PRIMITIVES - {"SumPool2d"}),
    substitutions={"SumPool2d": "AvgPool2d"},
)


@pytest.mark.parametrize("name", sorted(_PRESETS))
def test_reference_supports_every_node(name: str) -> None:
    """The reference target classifies every shipped node as supported."""
    matrix = capability_matrix.classify(_PRESETS[name], _REFERENCE)
    assert matrix.unsupported == ()
    assert matrix.substituted == ()
    assert matrix.supported
    assert matrix.deployable() is True
    assert matrix.counts()["total"] == len(matrix.supported)


def test_narrow_target_flags_only_pooling_nodes() -> None:
    """A target without pooling reports exactly the pool nodes unsupported."""
    matrix = capability_matrix.classify(_PRESETS["conv_net"], _NARROW)
    assert set(matrix.unsupported) == {"pool1", "pool2"}
    assert matrix.deployable() is False


def test_buckets_partition_the_node_set() -> None:
    """Supported, unsupported, and substituted partition the nodes."""
    target = _PRESETS["conv_net"]
    matrix = capability_matrix.classify(target, _NARROW)
    names = [name for name, _ in node_kinds(target)]
    buckets = (
        set(matrix.supported)
        | set(matrix.unsupported)
        | {item.name for item in matrix.substituted}
    )
    assert buckets == set(names)
    assert matrix.counts()["total"] == len(names)


def test_substituted_node_is_named_not_dropped() -> None:
    """A declared substitution lands in its own bucket with the form."""
    matrix = capability_matrix.classify(_PRESETS["conv_net"], _SUBSTITUTING)
    forms = {item.primitive: item.substitute for item in matrix.substituted}
    assert forms == {"SumPool2d": "AvgPool2d"}
    assert {item.name for item in matrix.substituted} == {"pool2"}
    assert matrix.unsupported == ()
    assert matrix.deployable() is True


def test_matrix_dict_is_json_serialisable() -> None:
    """The matrix exposes a JSON-serialisable form with counts."""
    matrix = capability_matrix.classify(_PRESETS["fc_legacy"], _NARROW)
    payload = matrix.to_dict()
    assert json.dumps(payload)
    total = len(node_kinds(_PRESETS["fc_legacy"]))
    assert payload["counts"]["total"] == total


def test_compare_targets_covers_every_registered_target() -> None:
    """Comparing across targets returns one JSON-able row per target."""
    rows = capability_matrix.compare_targets(_PRESETS["conv_net"])
    assert set(rows) == set(registry.target_names())
    assert json.dumps(rows)
