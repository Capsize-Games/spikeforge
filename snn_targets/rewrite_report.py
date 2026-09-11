"""The JSON-able report of a substitution rewrite.

The report partitions every node that the target does not support natively
into exactly one of three buckets, so nothing is silently dropped:

``applied``    a declared substitution with an executable rule was performed;
``skipped``    a substitution is declared but no rule could execute it;
``unfixable``  no substitution is declared for the node's primitive.

``rewritten`` is true when at least one substitution was applied. ``ready``
is true only when every node is either supported or successfully substituted,
so a caller can gate compilation on it. The optional ``drift`` section
quantifies how far the rewritten graph's trajectory moved from the original
when both were executed by the reference interpreter.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

#: Record list type: one flat JSON-able mapping per affected node.
Records = Tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class RewriteReport:
    """The applied/skipped/unfixable outcome of rewriting for one target."""

    target: str
    applied: Records
    skipped: Records
    unfixable: Records
    rewritten: bool
    drift: Optional[Mapping[str, Any]] = field(default=None)

    def ready(self) -> bool:
        """Return True when no node remains skipped or unfixable."""
        return not self.skipped and not self.unfixable

    def counts(self) -> Dict[str, int]:
        """Return the size of every rewrite bucket."""
        return {
            "applied": len(self.applied),
            "skipped": len(self.skipped),
            "unfixable": len(self.unfixable),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the report."""
        return {
            "target": self.target,
            "applied": [dict(item) for item in self.applied],
            "skipped": [dict(item) for item in self.skipped],
            "unfixable": [dict(item) for item in self.unfixable],
            "rewritten": self.rewritten,
            "ready": self.ready(),
            "counts": self.counts(),
            "drift": None if self.drift is None else dict(self.drift),
        }
