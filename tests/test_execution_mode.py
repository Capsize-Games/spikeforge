"""Tests for the ExecutionMode scaffold."""

from snn_interpreter.runtime.execution_mode import ExecutionMode


def test_both_modes_exist() -> None:
    """The educational and production members are defined."""
    assert ExecutionMode.EDUCATIONAL is not None
    assert ExecutionMode.PRODUCTION is not None


def test_modes_are_distinct() -> None:
    """The two modes are separate enum members."""
    assert ExecutionMode.EDUCATIONAL is not ExecutionMode.PRODUCTION
