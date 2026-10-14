"""Bundle fixtures shared by the ``spikeforge-serve`` app tests.

Both ``test_serve_app`` halves write the same small deterministic bundle
and sample the same raw inputs, so the writers live here instead of being
copied into each file.
"""

from pathlib import Path
from typing import Any, List

import torch

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.topology import registry

_TOPOLOGY = "fc_small"
_PARAMS = {"hidden": 5, "num_classes": 3}
_NUM_STEPS = 4


def _write_bundle(path: Path) -> None:
    """Write a small, deterministic ``.spkf`` bundle for the tests."""
    spec, module = registry.build_topology(_TOPOLOGY, dict(_PARAMS))
    module.eval()
    bundle = DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
            "topology": _TOPOLOGY,
            "topology_params": dict(_PARAMS),
            "encode_spec_version": ENCODE_SPEC_VERSION,
            "num_classes": _PARAMS["num_classes"],
            "label_map": {"0": "a", "1": "b", "2": "c"},
            "expected_metrics": {"test_accuracy": 0.9},
        },
        weights=module.state_dict(),
        encode_config=EncodeSpec(
            coding="latency", num_steps=_NUM_STEPS
        ).to_dict(),
    )
    bundle.save(str(path))


def _sample(seed: int) -> List[Any]:
    """Return one deterministic ``[1, 28, 28]`` raw sample as JSON."""
    torch.manual_seed(seed)
    return torch.rand(1, 28, 28).tolist()
