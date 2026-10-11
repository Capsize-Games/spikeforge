"""Raw-sample inference must preserve temporal and batch dimensions."""

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

from spikeforge.benchmark.serving import (
    ServingBenchmarkConfig,
    run_serving_benchmark,
)
from spikeforge.serving import bundle_manifest
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.serving.session import InferenceSession
from spikeforge.simulator.runner import run
from spikeforge.topology import registry
from spikeforge.topology.stage_module import StageModule

_CASES = [
    ("fc_legacy", {"hidden": 6, "num_classes": 3}),
    ("fc_small", {"hidden": 5, "num_classes": 3}),
    ("recurrent_net", {"hidden": 6, "num_classes": 3}),
    ("conv_net", {"channels": 2, "num_classes": 3}),
]


def _model(
    name: str, params: Dict[str, Any], steps: int = 5,
) -> Tuple[DeploymentBundle, StageModule]:
    """Return a seeded module and a bundle with deterministic encoding."""
    torch.manual_seed(0)
    spec, module = registry.build_topology(name, params)
    module.eval()
    return DeploymentBundle(
        manifest={
            "format": bundle_manifest.BUNDLE_FORMAT,
            "version": bundle_manifest.BUNDLE_VERSION,
            "spec": spec.to_dict(),
            "topology": name,
        },
        weights=module.state_dict(),
        encode_config=EncodeSpec(
            coding="latency", num_steps=steps,
        ).to_dict(),
    ), module


@pytest.mark.parametrize("name,params", _CASES)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("steps", [1, 5])
def test_raw_sample_matches_whole_tensor_simulation(
    name: str, params: Dict[str, Any], batch: int, steps: int,
) -> None:
    """Raw sample predictions retain batch shape and temporal dynamics."""
    bundle, module = _model(name, params, steps)
    session = InferenceSession.load(bundle)
    sample = torch.rand(batch, 1, 28, 28)
    if batch == 1:
        sample = sample[0]
    expected = run(module, session.encode(sample))
    actual = session.step_sample(sample)
    assert actual.steps == steps
    assert actual.mean_logits.shape == expected.logits.shape == (batch, 3)
    assert torch.allclose(actual.mean_logits, expected.logits, atol=1e-6)


@pytest.mark.parametrize("name,params", _CASES)
def test_raw_samples_preserve_state_across_encoded_trains(
    name: str, params: Dict[str, Any],
) -> None:
    """Each yielded sample prediction includes every carried timestep."""
    bundle, _module = _model(name, params)
    session = InferenceSession.load(bundle)
    manual = InferenceSession.load(bundle)
    samples = [torch.rand(1, 28, 28) for _ in range(2)]
    frames = torch.cat([manual.encode(sample) for sample in samples])
    expected = list(manual.run_stream(frames))[-1]
    predictions = list(session.run_samples(samples))
    assert [prediction.steps for prediction in predictions] == [5, 10]
    assert predictions[-1].logits.shape == (1, 3)
    assert torch.allclose(predictions[-1].class_totals, expected.class_totals)
    assert session.state() == manual.state()


@pytest.mark.parametrize("name,params", _CASES)
def test_individual_frame_after_raw_sample_keeps_normal_state(
    name: str, params: Dict[str, Any],
) -> None:
    """Raw samples and individual frames share one state shape."""
    bundle, _module = _model(name, params)
    session = InferenceSession.load(bundle)
    manual = InferenceSession.load(bundle)
    sample = torch.rand(1, 28, 28)
    frames = manual.encode(sample)
    list(manual.run_stream(frames))
    session.step_sample(sample)
    expected = manual.step(frames[0])
    actual = session.step(frames[0])
    assert actual.steps == expected.steps == 6
    assert actual.logits.shape == expected.logits.shape == (1, 3)
    assert torch.allclose(actual.class_totals, expected.class_totals)


@pytest.mark.parametrize("name,params", _CASES)
def test_serving_benchmark_accepts_each_sample_topology(
    name: str, params: Dict[str, Any], tmp_path: Path,
) -> None:
    """Cold start and timed calls use compatible encoded frame shapes."""
    bundle, _module = _model(name, params)
    path = tmp_path / "model.spkf"
    bundle.save(str(path))
    report = run_serving_benchmark(ServingBenchmarkConfig(
        bundle=str(path), calls=2, warmup=1,
    ))
    serving = report["results"][0]["serving"]
    assert serving["calls"] == 2
    assert serving["frames_per_call"] == 5
    assert serving["cold_start_ms"] > 0
