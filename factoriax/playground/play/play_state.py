"""Mutable UI state for the interactive play loop.

This dataclass holds all menu visibility, navigation, and recording
state. It is entirely separate from :class:`~factoriax.engine.state.EnvState`
and is never passed to JAX functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factoriax.engine.state import EnvState
    from factoriax.playground.ui.primitives import ClickRegion


@dataclass
class PlayState:
    """UI-only state managed by the play loop.

    Attributes:
        inventory_open: Whether the inventory/crafting menu is visible.
        achievement_open: Whether the achievement menu is visible.
        pause_open: Whether the pause menu is visible.
        help_open: Whether the help overlay is visible.
        machine_open: Whether the machine inspection menu is visible.
        welcome_open: Whether the one-time welcome screen is showing.
        victory_open: Whether the victory screen is showing.
        victory_shown: Whether the victory screen has been shown this session.
        achievement_scroll: Vertical pixel scroll offset in the achievement list.
        achievement_selection: Currently highlighted achievement row.
        pause_selection: Currently highlighted pause option (0-2).
        menu_focus: Which side of the inventory menu has focus.
        machine_tx: X tile coordinate of the inspected machine.
        machine_ty: Y tile coordinate of the inspected machine.
        machine_panel_active: Whether the machine panel (vs player panel) has focus.
        held_item: Item type currently held for swapping, or None.
        selected_item: Currently selected item type (machine for building).
        focused_machine_item: Currently focused machine item type.
        record_enabled: Whether trajectory recording is active.
        recorded_states: Captured EnvState snapshots for trajectory.
        recorded_actions: Captured action integers per step.
        recorded_rewards: Captured reward floats per step.
        frame_tick: Frame counter for animation timing.
    """

    # Menu visibility
    inventory_open: bool = False
    achievement_open: bool = False
    pause_open: bool = False
    help_open: bool = False
    machine_open: bool = False
    welcome_open: bool = True
    victory_open: bool = False
    victory_shown: bool = False

    # Menu navigation
    achievement_scroll: int = 0
    achievement_selection: int = 0
    pause_selection: int = 0
    menu_focus: str = "inventory"
    machine_tx: int = 0
    machine_ty: int = 0
    machine_panel_active: bool = True
    held_item: int | None = None
    selected_item: int = 4  # ItemType.MINER (first tool belt machine)
    focused_machine_item: int = 0
    selected_recipe: int = 0

    # Recording
    record_enabled: bool = False
    recorded_states: list[EnvState] = field(default_factory=list)
    recorded_actions: list[int] = field(default_factory=list)
    recorded_rewards: list[float] = field(default_factory=list)

    # UI hit regions from the last rendered frame
    click_regions: list[ClickRegion] = field(default_factory=list)

    # Frame state
    frame_tick: int = 0

    # Mouse-driven facing: last Direction the mouse pointed toward,
    # or 0 when unknown. Used to emit FACE_* only on change.
    mouse_facing: int = 0

    # Tile the mouse is hovering over, or (-1, -1) when off the world.
    hover_tile_x: int = -1
    hover_tile_y: int = -1
