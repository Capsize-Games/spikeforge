#!/usr/bin/env python3
"""Static core-boundary scan for the ``snn-interpreter`` distribution.

ARCH-0001 Phase 1c. The core distribution must install and import without any
of the forbidden root packages (see ``plans/arch-0001-core-boundary.md``). This
script enforces the *static* half of that decision: it parses every module the
core distribution ships and fails on

* a **module-level** ``import``/``from`` of a forbidden root, anywhere, and
* a **function-local** import of a forbidden root outside the enumerated
  lazy-shim allow-list.

The runtime half is the extended ``blocked-deps`` gate; the artifact half is
the ``headless`` CI job. This script is the cheap, fast, offline half.

Usage::

    python scripts/check_core_boundary.py [--quiet]

It exits 0 when the tree is clean and 1 (with a per-violation report)
otherwise. Each report line is ``path:line: kind import of forbidden root
'root'`` so a reviewer can jump straight to the offending statement.
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Set

#: Repository root (this file lives in ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Forbidden import roots, frozen for ``protocol_version`` 1.0 (13 tokens).
#:
#: The set mixes distribution names and import roots deliberately. Where they
#: differ: the ``lava-nc`` *distribution* is imported as ``lava``, so only
#: ``lava`` can ever appear in an ``import`` statement; ``huggingface_hub``,
#: ``pydantic``, ``fastapi``, ``uvicorn``, ``nir``, ``nirtorch``, ``onnx``,
#: ``onnxruntime``, ``tonic``, ``tensorboard``, and ``wandb`` use the same
#: token for both. ``FROZEN_BOUNDARY`` below records the full design list for
#: the ``--list`` output.
FORBIDDEN_ROOTS: Set[str] = frozenset(
    {
        "fastapi",
        "pydantic",
        "uvicorn",
        "huggingface_hub",
        "nir",
        "nirtorch",
        "onnx",
        "onnxruntime",
        "norse",
        "lava",
        "tonic",
        "tensorboard",
        "wandb",
    }
)

#: The design document's frozen boundary, including the ``lava-nc``
#: distribution token that never appears as an import root.
FROZEN_BOUNDARY: Sequence[str] = (
    "fastapi",
    "pydantic",
    "uvicorn",
    "huggingface_hub",
    "nir",
    "nirtorch",
    "onnx",
    "onnxruntime",
    "norse",
    "lava",
    "lava-nc",
    "tonic",
    "tensorboard",
    "wandb",
)

#: Sanctioned lazy shims (tier 3) that may import a forbidden SDK *inside a
#: function* and report availability instead of raising. Module-level imports
#: are never exempt, even in these modules.
#:
#: The design document names ``tracking.tensorboard_sink`` and
#: ``tracking.wandb_sink`` as the tracking exception; the concrete ``import
#: wandb`` lives in ``tracking.sink_probe``, whose own docstring calls it "the
#: only module ... that imports tensorboard or wandb". It is the same
#: sanctioned site, so it is listed here too.
ALLOWED_LOCAL_IMPORTERS: Set[str] = frozenset(
    {
        "snn_interpreter.nir_bridge.api",
        "snn_interpreter.onnx_bridge.api",
        "snn_interpreter.events.tonic_api",
        "snn_interpreter.hub.probe",
        "snn_interpreter.tracking.tensorboard_sink",
        "snn_interpreter.tracking.wandb_sink",
        "snn_interpreter.tracking.sink_probe",
    }
)

#: Directories the core distribution never ships, so they are never scanned.
#: Mirrors the ``exclude`` list in ``packages/snn-interpreter/pyproject.toml``
#: (``server*``, ``tests*``, ``snn_targets*``, ``snn_hub*``).
EXCLUDED_PREFIXES = ("server", "tests", "snn_targets", "snn_hub")

#: Package directories the core distribution ships, relative to the repo root.
SCAN_DIRS = ("snn_interpreter",)

#: Top-level single-file modules the core distribution ships.
SCAN_FILES = ("main.py", "main_encodings.py")


class Violation(NamedTuple):
    """A single forbidden import found in a scanned module."""

    path: Path
    line: int
    kind: str
    root: str
    statement: str


class _Visitor(ast.NodeVisitor):
    """Collect forbidden imports, separating module- from function-level."""

    def __init__(
        self, module: str, path: Path, violations: List[Violation]
    ) -> None:
        """Record the module name, its path, and the shared result list."""
        self._module = module
        self._path = path
        self._violations = violations
        self._function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Descend into a sync function body, marking imports as local."""
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Descend into an async function body (same rule as sync)."""
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        """Check every binding in ``import a, b``."""
        for alias in node.names:
            self._check(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check the module named by ``from x.y import ...``."""
        if node.level:
            return
        if node.module:
            self._check(node.module, node.lineno)

    def _check(self, dotted: str, line: int) -> None:
        """Record a violation unless this is a sanctioned lazy import."""
        root = dotted.split(".", 1)[0]
        if root not in FORBIDDEN_ROOTS:
            return
        if self._function_depth:
            if self._module in ALLOWED_LOCAL_IMPORTERS:
                return
            kind = "function-local"
        else:
            kind = "module-level"
        self._violations.append(
            Violation(self._path, line, kind, root, dotted)
        )


def _module_name(path: Path) -> str:
    """Return the dotted module name for a scanned file."""
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_excluded(relative: Path) -> bool:
    """Return True when a repo-relative path is outside the core ship set."""
    return any(
        part.startswith(EXCLUDED_PREFIXES) for part in relative.parts
    )


def iter_modules() -> List[Path]:
    """Return the ``.py`` files the core distribution ships, sorted."""
    modules: List[Path] = []
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if _is_excluded(path.relative_to(REPO_ROOT)):
                continue
            modules.append(path)
    for name in SCAN_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            modules.append(path)
    return modules


def scan(paths: Sequence[Path]) -> List[Violation]:
    """Parse ``paths`` and return every core-boundary violation found."""
    violations: List[Violation] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        _Visitor(_module_name(path), path, violations).visit(tree)
    return violations


def format_violation(violation: Violation) -> str:
    """Render one violation as a ``path:line`` report line."""
    relative = violation.path.relative_to(REPO_ROOT)
    return (
        f"{relative}:{violation.line}: {violation.kind} import of "
        f"forbidden root '{violation.root}' ({violation.statement})"
    )


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fail when a core-distribution module imports a forbidden root."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing on success (violations are always reported)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the frozen boundary and the lazy-shim allow-list",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the scan and return a process exit code."""
    args = _parse_args(argv)

    if args.list:
        print("forbidden boundary (design): " + ", ".join(FROZEN_BOUNDARY))
        print("forbidden import roots (scanned): " + ", ".join(
            sorted(FORBIDDEN_ROOTS)
        ))
        print("lazy-shim allow-list: " + ", ".join(
            sorted(ALLOWED_LOCAL_IMPORTERS)
        ))
        return 0

    modules = iter_modules()
    violations = scan(modules)
    if violations:
        for violation in violations:
            print(format_violation(violation), file=sys.stderr)
        print(
            f"core-boundary: {len(violations)} violation(s) across "
            f"{len(modules)} module(s)",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print(
            f"core-boundary: OK - {len(modules)} module(s) scanned, "
            "no forbidden imports"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
