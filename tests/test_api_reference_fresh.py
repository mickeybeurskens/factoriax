"""Verify ``docs/api-reference.md`` matches what the generator produces.

The committed reference is the source of truth shown to readers, and
the generator at ``scripts/generate_api_reference.py`` is what we
use to produce it. If the two drift — typically because someone
edited ``factoriax/__init__.py`` or ``factoriax/_stability.py``
without re-running the script — this test fails. Re-running the
script and committing the diff is the documented fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``scripts/`` importable so the generator can be called in-process.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from generate_api_reference import render_reference  # noqa: E402

_COMMITTED = _REPO / "docs" / "api-reference.md"


def test_api_reference_matches_committed() -> None:
    """The committed markdown is byte-identical to the generator output."""
    expected = render_reference()
    actual = _COMMITTED.read_text()
    assert expected == actual, (
        "docs/api-reference.md is stale. Re-run "
        "`uv run python scripts/generate_api_reference.py` and commit."
    )
