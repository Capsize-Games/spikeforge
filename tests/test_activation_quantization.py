"""Activation/membrane quantization: opt-in, recorded, and honest."""

import json
from typing import Any

import torch

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.session import InferenceSession
from spikeforge.simulator.state import initial_state
from spikeforge.simulator.step_stages import step_stages
from spikeforge.topology import registry
from spikeforge_targets.activation_quant import (
    NO_ACTIVATION_SCHEME,
    SCHEMES,
    ActivationQuantizer,
    Calibration,
    calibrate,
    scheme_for,
    scheme_names,
)

#: Largest logit difference the 8-bit activation grid may introduce.
TOLERANCE = 5e-2

_PARAMS = {"hidden": 5, "num_classes": 3}


def _bundle() -> Any:
    """Return an in-memory fc_small bundle with seeded weights."""
    torch.manual_seed(0)
    spec, module = registry.build_topology("fc_small", dict(_PARAMS))
    return DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=module.state_dict(),
    )


def _frames() -> Any:
    """Return a fixed batch of input frames."""
    torch.manual_seed(1)
    return [torch.rand(2, 28, 28) for _ in range(3)]


def test_default_session_is_unchanged_by_the_hook() -> None:
    """A plain session has no post-step and matches a direct step exactly."""
    bundle = _bundle()
    session = InferenceSession.load(bundle)
    assert session.post_step is None
    frame = _frames()[0]
    prediction = session.step(frame)
    module = bundle.build_module()
    outputs, _state = step_stages(
        module, frame, dict(initial_state(session.spec, frame))
    )
    assert torch.equal(prediction.logits, outputs[session.spec.output])


def test_identity_post_step_is_byte_identical() -> None:
    """An identity hook leaves every step exactly as the default path."""
    bundle = _bundle()
    frame = _frames()[0]
    plain = InferenceSession.load(bundle).step(frame)
    hooked = InferenceSession.load(
        bundle, post_step=lambda outputs, state: (outputs, state)
    ).step(frame)
    assert torch.equal(plain.logits, hooked.logits)
    assert torch.equal(plain.class_totals, hooked.class_totals)


def test_activation_quantizer_records_ranges_and_error() -> None:
    """An opt-in quantizer applies, records layers, and stays close."""
    frames = _frames()
    reference = [
        InferenceSession.load(_bundle()).step(f) for f in frames
    ]
    quantizer = ActivationQuantizer("activation_membrane_int8")
    session = InferenceSession.load(_bundle(), post_step=quantizer)
    got = [session.step(f) for f in frames]
    report = quantizer.report()
    assert report.applied is True
    assert report.scheme == "activation_membrane_int8"
    assert report.target == "both"
    assert report.counts()["layers"] > 0
    assert report.counts()["steps"] == len(frames)
    assert all(item["max_abs"] >= 0.0 for item in report.layers)
    assert json.dumps(report.to_dict())
    for observed, expected in zip(got, reference):
        assert torch.allclose(
            observed.logits, expected.logits, atol=TOLERANCE
        )


def test_unknown_scheme_is_refused_honestly() -> None:
    """An unsupported scheme is named and never approximated."""
    quantizer = ActivationQuantizer("activation_int4")
    assert quantizer.applied is False
    frame = _frames()[0]
    plain = InferenceSession.load(_bundle()).step(frame)
    session = InferenceSession.load(_bundle(), post_step=quantizer)
    hooked = session.step(frame)
    report = quantizer.report()
    assert report.applied is False
    assert "activation_int4" in report.reason
    assert torch.equal(hooked.logits, plain.logits)


def test_none_scheme_is_an_explicit_noop() -> None:
    """``none`` is reported as disabled rather than dropped."""
    quantizer = ActivationQuantizer("none")
    InferenceSession.load(_bundle(), post_step=quantizer).step(_frames()[0])
    report = quantizer.report()
    assert report.applied is False
    assert report.reason == NO_ACTIVATION_SCHEME


def test_each_scheme_targets_the_declared_tensors() -> None:
    """Activation-only and membrane-only runs record disjoint layer kinds."""
    frame = _frames()[0]
    activation = ActivationQuantizer("activation_int8")
    InferenceSession.load(_bundle(), post_step=activation).step(frame)
    kinds = {item["kind"] for item in activation.report().layers}
    assert kinds == {"activation"}

    membrane = ActivationQuantizer("membrane_int8")
    InferenceSession.load(_bundle(), post_step=membrane).step(frame)
    kinds = {item["kind"] for item in membrane.report().layers}
    assert kinds == {"membrane"}


def test_calibration_hook_is_recorded_in_the_report() -> None:
    """A calibration dataset pins the grid and appears in the report."""
    torch.manual_seed(2)
    observed = [
        ("fc1", torch.randn(2, 5)),
        ("fc2", torch.randn(2, 3)),
    ]
    calibration = calibrate(observed, bits=8, target="activation")
    assert isinstance(calibration, Calibration)
    assert calibration.samples == 2
    assert calibration.bound("fc1") is not None

    quantizer = ActivationQuantizer(
        "activation_int8", calibration=calibration
    )
    InferenceSession.load(_bundle(), post_step=quantizer).step(_frames()[0])
    report = quantizer.report()
    assert report.calibration is not None
    assert report.calibration["samples"] == 2
    assert "fc1" in report.calibration["ranges"]


def test_scheme_registry_is_documented_and_lookup_is_total() -> None:
    """The registry lists its schemes and an unsupported one resolves None."""
    assert set(scheme_names()) == set(SCHEMES)
    assert scheme_for("activation_membrane_int8") is not None
    assert scheme_for("does_not_exist") is None
    assert "none" in SCHEMES


def test_report_is_json_serialisable() -> None:
    """A refusal report is JSON-able like the weight quantization report."""
    quantizer = ActivationQuantizer("activation_int4")
    payload = quantizer.report().to_dict()
    assert json.dumps(payload)
    assert payload["applied"] is False
