"""Test that the play UI never edits the environment state itself.

The engine owns the dynamics. The play UI turns input into an ``Action`` and
hands it to :func:`factoriax.engine.step.step`, so a human and a policy move
the world through the same code. A direct write to ``EnvState`` from the UI
gives the human a rule that the policy does not have.

This test reads the play sources and rejects the two ways a caller can write
to a flax struct: ``state.replace(...)`` and ``state.<field>.at[...].set(...)``.
The editor is not covered, because editing state is what the editor is for.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PLAY_DIR = Path(__file__).resolve().parents[2] / "factoriax" / "playground" / "play"

# ``selected_player`` says which player an action applies to. It routes the
# action and does not advance the world. The engine offers no action for it:
# EnvState documents that the caller sets this field.
_ALLOWED_REPLACE_FIELDS = frozenset({"selected_player"})


def _play_sources() -> list[Path]:
    return sorted(p for p in _PLAY_DIR.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _play_sources(), ids=lambda p: p.name)
def test_no_state_replace(path: Path) -> None:
    """``state.replace(...)`` rewrites the world outside the engine."""
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [
        f"{path.name}:{node.lineno} replace({', '.join(sorted(fields))})"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        for fields in [{kw.arg for kw in node.keywords if kw.arg}]
        if not fields <= _ALLOWED_REPLACE_FIELDS
    ]
    assert offenders == []


@pytest.mark.parametrize("path", _play_sources(), ids=lambda p: p.name)
def test_no_jax_in_place_update(path: Path) -> None:
    """``.at[...].set(...)`` writes into an engine array outside the engine."""
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [
        f"{path.name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "at"
    ]
    assert offenders == []
