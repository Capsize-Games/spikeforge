"""Module entry point: ``python -m spikeforge.benchmark``.

The parser and handlers live in
:mod:`spikeforge.benchmark.cli`; this module only exposes ``main`` so
the module runner and the ``spikeforge-benchmark`` console script share
one path.
"""

import sys

from spikeforge.benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
