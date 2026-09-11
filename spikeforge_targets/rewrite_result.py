"""The target-ready graph produced by a substitution rewrite.

``graph`` is a fresh NIR graph with every declared substitution applied, and
``report`` names what changed, what could not, and (when a spike fixture was
supplied) the measured post-rewrite drift. The original graph is never
mutated, so a caller can hold both the before and after renderings.
"""

from dataclasses import dataclass
from typing import Any

from spikeforge_targets.rewrite_report import RewriteReport


@dataclass(frozen=True)
class RewriteResult:
    """A rewritten graph paired with the report describing the rewrite."""

    graph: Any
    report: RewriteReport

    def ready(self) -> bool:
        """Return True when the rewritten graph has no remaining gap."""
        return self.report.ready()
