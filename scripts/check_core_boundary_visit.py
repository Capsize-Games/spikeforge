"""The AST visitor that collects forbidden imports per module."""

import ast
from pathlib import Path
from typing import List

from check_core_boundary_rules import (
    ALLOWED_LOCAL_IMPORTERS,
    FORBIDDEN_ROOTS,
    Violation,
)


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
