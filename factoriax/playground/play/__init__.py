"""FactoriaX Play — human-facing gameplay interface.

This subpackage contains pygame-based UI components for interactive play.
It is intentionally separated from the core RL environment so that the
training pipeline never pulls in pygame as a dependency.

Quick start
-----------
>>> python -m factoriax.playground.play

Or import directly:

>>> from factoriax.playground.play.main import main
>>> main()

Submodules
----------
- ``main`` — Game loop and entry point
- ``game_ui`` — Reusable GameUI component for menus and input dispatch
- ``ui`` — Menu rendering (inventory, achievements, pause)
"""

from factoriax.playground.play.game_ui import GameUI, GameUIResult
from factoriax.playground.play.ui import (
    render_achievement_menu,
    render_inventory_menu,
    render_pause_menu,
)

__all__ = [
    "GameUI",
    "GameUIResult",
    "render_achievement_menu",
    "render_inventory_menu",
    "render_pause_menu",
]
