"""Deterministic, offline spike fixtures shaped for a topology's input.

The deployment commands (``deploy``, ``rewrite``, ``run``, ``roundtrip``) and
the ``validate`` sequence path all need a spike volume shaped exactly like a
topology's input stage, generated without touching a dataset. A feature or
convolutional topology takes a flat ``[T, B, F]`` volume, reshaped to
``[T, B, C, H, W]`` for a spatial input; a sequence topology takes a
``[T, B, L, D]`` frame, or ``[T, B, L]`` integer tokens for an ``embedding``
entry. Nothing here reads or downloads data, so every fixture is reproducible
offline -- weights included: the module is built under the fixture's seed, not
merely the spikes that drive it.
"""

from typing import Any, Mapping, Tuple

import torch

from spikeforge.data import sequence_source
from spikeforge.simulator import input_shape
from spikeforge.topology.registry import (
    build_topology,
    is_sequence_topology,
    resolved_params,
)
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule

#: Default synthetic fixture geometry shared by the deployment commands.
STEPS = 8
BATCH = 2
FEATURES = 784
SEED = 0

Fixture = Tuple[TopologySpec, StageModule, torch.Tensor]


def _seeded_topology(
    topology: str, seed: int
) -> Tuple[TopologySpec, StageModule]:
    """Return ``topology`` built with reproducible weights.

    ``build_topology`` draws its initialisation from the global torch RNG, so
    the seed has to be set before the build and not merely before the input
    volume. Seeding only the input left every fixture with fresh weights on
    each call, which made a deployment report's weight ranges, quantization
    error and drift magnitudes differ from one invocation to the next while
    presenting as a deterministic offline fixture.
    """
    torch.manual_seed(seed)
    return build_topology(topology)


def _token_frames(
    spec: TopologySpec,
    module: StageModule,
    resolved: Mapping[str, Any],
    steps: int,
    batch: int,
    seed: int,
) -> Fixture:
    """Return a ``[T, B, L]`` integer-token fixture for an embedding entry."""
    length = int(resolved.get("seq_length", sequence_source.DEFAULT_LENGTH))
    vocab = int(resolved.get("vocab", sequence_source.DEFAULT_VOCAB))
    source = sequence_source.SequenceSource(length, vocab, seed=seed)
    frames = [
        source.tokens().unsqueeze(0).repeat(batch, 1) for _ in range(steps)
    ]
    return spec, module, torch.stack(frames)


def _dense_frames(
    spec: TopologySpec,
    module: StageModule,
    resolved: Mapping[str, Any],
    steps: int,
    batch: int,
    seed: int,
) -> Fixture:
    """Return a ``[T, B, L, D]`` feature fixture for a sequence preset."""
    length = int(resolved.get("seq_length", sequence_source.DEFAULT_LENGTH))
    features = int(resolved.get("features", sequence_source.DEFAULT_FEATURES))
    frames = sequence_source.random_frames(
        steps, batch, length, features, seed
    )
    return spec, module, frames


def sequence_input(
    topology: str, steps: int, batch: int, seed: int
) -> Fixture:
    """Return a ``[T, B, L, D]`` (or token) fixture for a sequence preset."""
    resolved = resolved_params(topology)
    spec, module = _seeded_topology(topology, seed)
    if spec.stage(spec.input).kind == "embedding":
        return _token_frames(spec, module, resolved, steps, batch, seed)
    return _dense_frames(spec, module, resolved, steps, batch, seed)


def synthetic_input(
    topology: str,
    steps: int = STEPS,
    batch: int = BATCH,
    seed: int = SEED,
) -> Fixture:
    """Return ``(spec, module, spikes)`` shaped for ``topology``."""
    if is_sequence_topology(topology):
        return sequence_input(topology, steps, batch, seed)
    spec, module = _seeded_topology(topology, seed)
    # Re-seed so the input volume is a function of the seed and its shape
    # alone, never of how much RNG the initialisation above happened to
    # consume: a preset that gains a stage must not silently change the
    # spikes every other preset's fixture is compared on.
    torch.manual_seed(seed)
    flat = torch.rand(steps, batch, FEATURES)
    return spec, module, input_shape.to_input_shape(flat, spec)
