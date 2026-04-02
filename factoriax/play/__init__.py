"""FactoriaX Play — human-facing gameplay interface.

This subpackage contains pygame-based UI components for interactive play.
It is intentionally separated from the core RL environment so that the
training pipeline never pulls in pygame as a dependency.

Quick start
-----------
>>> python -m factoriax.play

Or import directly:

>>> from factoriax.play import main
>>> main()

Submodules
----------
- ``main`` — Game loop and entry point
- ``ui`` — Menu rendering (inventory, achievements, pause)
"""

from factoriax.play.main import main
from factoriax.play.ui import (
    render_achievement_menu,
    render_inventory_menu,
    render_pause_menu,
)
from factoriax.ui.primitives import ClickRegion

__all__ = [
    "main",
    "ClickRegion",
    "render_achievement_menu",
    "render_inventory_menu",
    "render_pause_menu",
]
