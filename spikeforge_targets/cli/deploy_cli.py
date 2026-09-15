"""The ``deploy`` subcommand: one topology's deployment report for a target.

The report is :func:`~spikeforge_targets.report.deployment_report` shaped
for the CLI: a dataset sample or a deterministic synthetic fixture drives the
validation, substitution, and quantization sections, and the exit status
follows ``deployable`` so the command works as a CI gate. The
``--activation-quantization`` flag opts the quantization drift check into a
simulated activation/membrane scheme; without it the target's declared
scheme applies, which is ``none`` on every shipped target.
"""

import argparse
import json
from typing import Any, Dict, Optional

from spikeforge.cli import fixture
from spikeforge_targets.report import deployment_report

#: Target used for a deployment report when the caller names none.
DEFAULT_TARGET = "reference"
#: Synthetic input shape shared by the deploy, round-trip and ingest commands.
STEPS = 8
BATCH = 2
SEED = 0


def deploy_report(
    topology: str,
    target: str = DEFAULT_TARGET,
    dataset: Optional[str] = None,
    sample: int = 0,
    activation: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the deployment report for ``topology`` against ``target``.

    A ``dataset`` selects the sample whose spikes drive the validation and
    substitution sections; without it a deterministic synthetic fixture is
    used, so the report still carries the executed substitution view while
    staying offline. ``activation`` names the simulated scheme for the
    quantization drift check, overriding the target's declared one.
    """
    if dataset is None:
        spec, module, spikes = fixture.synthetic_input(
            topology, STEPS, BATCH, SEED
        )
    else:
        from spikeforge.cli import verify

        spec, module, spikes = verify.sample_input(topology, dataset, sample)
    return deployment_report(
        spec, target, module=module, spikes=spikes, activation=activation
    )


def deploy_exit(report: Dict[str, Any]) -> int:
    """Return the process status for a deployment report."""
    return 0 if report["deployable"] else 1


def _run_deploy(args: argparse.Namespace) -> int:
    """Print a deployment report and return its usability status."""
    report = deploy_report(
        args.topology, args.target, args.dataset, args.sample, args.activation
    )
    print(json.dumps(report, indent=2))
    return deploy_exit(report)


def add_deploy_parser(subs: Any) -> None:
    """Register the ``deploy`` subcommand on ``subs``."""
    deploy = subs.add_parser("deploy", help="report a topology's target fit")
    deploy.add_argument("--topology", default="conv_net")
    deploy.add_argument("--target", default=DEFAULT_TARGET)
    deploy.add_argument("--dataset", default=None)
    deploy.add_argument("--sample", type=int, default=0)
    deploy.add_argument(
        "--activation-quantization",
        dest="activation",
        default=None,
        metavar="SCHEME",
        help=(
            "simulate this activation/membrane scheme in the quantization "
            "drift check (for example activation_membrane_int8); the "
            "default is the target's declared scheme"
        ),
    )
    deploy.set_defaults(handler=_run_deploy)
