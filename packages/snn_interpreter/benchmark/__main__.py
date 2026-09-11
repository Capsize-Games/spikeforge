"""Module entry point: ``python -m snn_interpreter.benchmark``.

The parser and handlers live in
:mod:`snn_interpreter.benchmark.cli`; this module only exposes ``main`` so
the module runner and the ``snn-benchmark`` console script share one path.
"""

import sys

from snn_interpreter.benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
