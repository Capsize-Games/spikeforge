"""Deterministic bundles and frames for the C-backend tests.

The fixture is a small ``conv_net`` (1x8x8 in, 3 classes out) with
seeded positive-leaning weights driving real spikes — centered
random weights mostly cancel out and read all zeros, which a broken
port would match by accident. The readout rows are gainscaled per
class so each class spikes on different steps; without that every
class fires in lockstep and a broadcast bug would still match.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.topology import registry
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage

#: Fixture topology size: 1x8x8 input (each side >= 4 for the pools).
_PARAMS: Dict[str, Any] = {
    "in_channels": 1,
    "channels": 2,
    "input_size": 8,
    "num_classes": 3,
}
#: Seeds pinning the weights and the frames, respectively.
_WEIGHT_SEED = 0
_FRAME_SEED = 1
#: Steps per stream and the per-step firing probability.
_STEPS = 6
_FIRING_RATE = 0.5
#: Per-class readout gains; see the module docstring for why.
_READOUT_GAINS = (0.4, 1.0, 1.8)


def _randomize(module: Any) -> None:
    """Replace every parameter with seeded positive-leaning values."""
    generator = torch.Generator().manual_seed(_WEIGHT_SEED)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(
                torch.rand(param.shape, generator=generator) * 0.15
            )
        gains = torch.tensor(_READOUT_GAINS).unsqueeze(1)
        module.fc.weight.mul_(gains)


def _conv_bundle() -> DeploymentBundle:
    """Return the deterministic in-memory conv_net fixture bundle.

    The bundle pins a rate-coded 8x8 encode spec so raw samples
    encode deterministically alongside the pre-encoded frames.
    """
    torch.manual_seed(_WEIGHT_SEED)
    spec, module = registry.build_topology("conv_net", dict(_PARAMS))
    module.eval()
    _randomize(module)
    encode = EncodeSpec(coding="rate", num_steps=_STEPS, input_size=(8, 8))
    return DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=dict(module.state_dict()),
        encode_config=encode.to_dict(),
    )


def _fc_bundle() -> DeploymentBundle:
    """Return a small in-memory ``fc_small`` bundle."""
    spec, module = registry.build_topology(
        "fc_small", {"hidden": 5, "num_classes": 3}
    )
    return DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=dict(module.state_dict()),
    )


def _linear_bundle() -> DeploymentBundle:
    """Return a bundle whose graph holds C-rejected ``Linear`` nodes."""
    torch.manual_seed(0)
    params = {"in_features": 4, "out_features": 3, "bias": False}
    spec = chain([Stage("fc1", "linear", params)])
    module = build_module(spec)
    return DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=dict(module.state_dict()),
    )


def _spike_frames(steps: int = _STEPS) -> torch.Tensor:
    """Return a seeded ``[T, 1, 1, 8, 8]`` spike train for the fixture."""
    generator = torch.Generator().manual_seed(_FRAME_SEED)
    return (
        torch.rand(steps, 1, 1, 8, 8, generator=generator)
        < _FIRING_RATE
    ).float()


def _write_bundle(tmp_path: Path, bundle: DeploymentBundle) -> str:
    """Save ``bundle`` under ``tmp_path``; return its path."""
    path = str(tmp_path / "model.spkf")
    bundle.save(path)
    return path


def _write_request(tmp_path: Path, frames: List[Any]) -> str:
    """Write an encoded one-request file; return its path."""
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"frames": frames, "encoded": True}))
    return str(path)
