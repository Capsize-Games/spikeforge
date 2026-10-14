#!/usr/bin/env python3
"""Regenerate the encoding golden vectors from the torch implementation.

The fixtures pin the exact input/output behavior of the four spike
codings plus the streaming pre-steps, so a future C encoder has an
oracle to verify against (issue #160). They record current behavior,
bugs included: if a regenerated fixture differs, the encoder math or
the seeding order changed and the change needs its own review.

Usage::

    PYTHONPATH=. python scripts/generate_encoding_goldens.py
    PYTHONPATH=. python scripts/generate_encoding_goldens.py --out DIR
"""

import argparse
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import snntorch
import torch
from encoding_golden_format import envelope, payload, write_cases

from spikeforge.serving import preprocess
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.streaming.encoding import delta_over_window
from spikeforge.streaming.window_spec import WindowSpec

#: Pinned seed for the stochastic codings (rate, random).
SEED: int = 1234
#: Time steps for every coding fixture; small enough to stay readable.
NUM_STEPS: int = 8

#: Fixed image pixels, exactly representable in float32 and in [0, 1]
#: so the latency coding accepts them.
PIXELS: List[float] = [
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    1.0,
]

_SEED_STEP = (
    f"torch.manual_seed({SEED}) immediately before dispatch, "
    "then SpikeEncoder.encode"
)

#: The four coding fixtures: name, spec overrides, procedure words.
CODINGS: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
    (
        "rate",
        {"coding": "rate", "num_steps": NUM_STEPS, "random_seed": SEED},
        f"preprocess.encode with random_seed={SEED}: {_SEED_STEP}.",
    ),
    (
        "latency",
        {"coding": "latency", "num_steps": NUM_STEPS},
        "preprocess.encode; deterministic, no RNG is consumed.",
    ),
    (
        "delta",
        {"coding": "delta", "num_steps": NUM_STEPS},
        "preprocess.encode; deterministic, no RNG is consumed.",
    ),
    (
        "random",
        {"coding": "random", "num_steps": NUM_STEPS, "random_seed": SEED},
        f"preprocess.encode with random_seed={SEED}: {_SEED_STEP}, "
        "then the encoder re-seeds with the same seed before its "
        "torch.rand + bernoulli draws.",
    ),
)


def _commit() -> str:
    """Return the current HEAD sha, or ``"unknown"`` off a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _provenance() -> Dict[str, str]:
    """Return the generator provenance block for every case."""
    return {
        "script": "scripts/generate_encoding_goldens.py",
        "spikeforge_commit": _commit(),
        "torch": torch.__version__,
        "snntorch": snntorch.__version__,
        "python": platform.python_version(),
    }


def _image() -> torch.Tensor:
    """Return the fixed ``[1, 1, 2, 4]`` fixture image."""
    return torch.tensor(PIXELS, dtype=torch.float32).reshape(1, 1, 2, 4)


def _coding_case(
    name: str, spec: Dict[str, Any], procedure: str,
    provenance: Dict[str, str],
) -> Dict[str, Any]:
    """Encode the fixture image under ``spec`` and return the case."""
    resolved = EncodeSpec.from_mapping(spec)
    spikes = preprocess.encode(_image(), resolved)
    case = envelope(name, f"{name} coding of the fixture image.",
                    provenance)
    case["spec"] = resolved.to_dict()
    case["spec_digest"] = resolved.digest()
    case["procedure"] = procedure
    case["input"] = payload(_image())
    case["output"] = payload(spikes)
    return case


def _coding_cases(provenance: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return the golden case for each of the four codings."""
    return [
        _coding_case(name, spec, procedure, provenance)
        for name, spec, procedure in CODINGS
    ]


def _delta_windows() -> torch.Tensor:
    """Return the fixed ``[1, 4, 3]`` first-difference input window."""
    return torch.tensor(
        [
            [
                [0.0, 0.5, 1.0],
                [0.25, 0.75, 1.5],
                [1.0, 0.125, 2.0],
                [0.5, 0.5, 0.5],
            ]
        ],
        dtype=torch.float32,
    )


def _delta_case(provenance: Dict[str, str]) -> Dict[str, Any]:
    """Return the ``delta_over_window`` golden case."""
    windows = _delta_windows()
    case = envelope(
        "delta_over_window",
        "First-difference of one [1, 4, 3] window along its time axis.",
        provenance,
    )
    case["procedure"] = (
        "spikeforge.streaming.encoding.delta_over_window; "
        "deterministic, no RNG is consumed."
    )
    case["input"] = payload(windows)
    case["output"] = payload(delta_over_window(windows))
    return case


def _normalize_spec() -> WindowSpec:
    """Return the frozen z-score spec of the normalize fixture."""
    return WindowSpec(
        length=4,
        stride=2,
        channels=2,
        channel_names=("a", "b"),
        mean=(0.5, -1.0),
        std=(0.25, 2.0),
    )


def _normalize_windows() -> torch.Tensor:
    """Return the fixed ``[1, 4, 2]`` z-score input windows."""
    return torch.tensor(
        [[[0.5, -1.0], [0.75, 1.0], [1.0, 3.0], [0.0, -5.0]]],
        dtype=torch.float32,
    )


def _normalize_case(provenance: Dict[str, str]) -> Dict[str, Any]:
    """Return the ``WindowSpec.normalize`` golden case."""
    spec = _normalize_spec()
    windows = _normalize_windows()
    case = envelope(
        "window_normalize",
        "Z-scoring of one [1, 4, 2] window batch with a frozen spec.",
        provenance,
    )
    case["spec"] = spec.to_dict()
    case["spec_digest"] = spec.digest()
    case["procedure"] = (
        "WindowSpec.normalize; deterministic, no RNG is consumed."
    )
    case["input"] = payload(windows)
    case["output"] = payload(spec.normalize(windows))
    return case


def generate() -> List[Dict[str, Any]]:
    """Return every golden case in a stable order."""
    provenance = _provenance()
    cases = _coding_cases(provenance)
    cases.append(_delta_case(provenance))
    cases.append(_normalize_case(provenance))
    return cases


def main(argv: Optional[List[str]] = None) -> int:
    """Generate the fixtures; ``--out`` overrides the default directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/encoding_goldens"),
        help="directory receiving the <name>.json fixtures",
    )
    args = parser.parse_args(argv)
    for path in write_cases(generate(), args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
