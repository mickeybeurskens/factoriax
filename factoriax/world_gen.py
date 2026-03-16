"""Backward-compatibility shim — world generation has moved to ``levels``.

Import :func:`factoriax.levels.generate_state` directly in new code.
"""

from factoriax.levels import generate_state as generate_world

__all__ = ["generate_world"]
