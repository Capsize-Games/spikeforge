"""Generate a deployment report and run the reference backend.

Classifies the ``conv_net`` topology against the always-available
``reference`` target and against an SDK-gated target (``xylo``), then
compiles and runs the reference backend and prints its comparison to the
independent interpreter.

Run from the repository root::

    venv/bin/python examples/06_deploy_and_run_backend.py

A target whose SDK is absent is reported honestly as unavailable (with a
named reason), never silently treated as ready. The equivalent shell
commands are ``snn-verify deploy --topology conv_net --target reference``
and ``snn-verify run --topology conv_net --target reference``.
"""

from typing import Any, Dict

from snn_interpreter.cli import fixture
from snn_interpreter.nir_bridge import to_nir
from snn_interpreter.targets.backends import compile_run
from snn_interpreter.targets.report import deployment_report

#: Topology exercised by this example.
TOPOLOGY = "conv_net"


def report_for(target: str) -> Dict[str, Any]:
    """Return a deployment report for ``target`` on the shared fixture."""
    spec, module, spikes = fixture.synthetic_input(TOPOLOGY, 4, 1, 0)
    return deployment_report(spec, target, module=module, spikes=spikes)


def main() -> int:
    """Print deployment reports and run the reference backend."""
    for target in ("reference", "xylo"):
        report = report_for(target)
        print(
            f"{target} deployable={report['deployable']} "
            f"counts={report['nodes']['counts']}"
        )
    spec, module, spikes = fixture.synthetic_input(TOPOLOGY, 4, 1, 0)
    result = compile_run("reference", to_nir(spec, module), spikes)
    compare = dict(result.compare or {})
    print("reference status:", result.status)
    print("readout max_abs:", (compare.get("readout") or {}).get("max_abs"))
    print(
        "spike agreement:",
        (compare.get("spikes") or {}).get("agreement"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
