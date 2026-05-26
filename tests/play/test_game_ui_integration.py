"""Integration tests: every GameUI handler with a real EnvState.

Each test creates a GameUI, sets up the required play state flags,
sends a pygame event, and verifies the result does not crash and
returns a sensible GameUIResult. This catches stale field references
that unit tests miss because they only exercise isolated paths.
"""

from __future__ import annotations

import jax.numpy as jnp
import pygame
import pytest

from factoriax.config import (
    build_key_lookup,
    default_keyboard,
)
from factoriax.constants import (
    Action,
    Direction,
    ItemType,
    Machine,
)
from factoriax.play.game_ui import GameUI, GameUIResult
from factoriax.state import EnvParams


@pytest.fixture
def params() -> EnvParams:
    """Small environment parameters."""
    return EnvParams(map_width=8, map_height=8, num_players=2)


@pytest.fixture
def kb_lookup():
    """Default keyboard lookup."""
    return build_key_lookup(default_keyboard())


@pytest.fixture
def ui(params, kb_lookup) -> GameUI:
    """A GameUI with welcome screen dismissed."""
    g = GameUI(params, kb_lookup, welcome_open=False)
    return g


def _key(key: int, mod: int = 0) -> pygame.event.Event:
    """Create a KEYDOWN event."""
    return pygame.event.Event(
        pygame.KEYDOWN,
        key=key,
        mod=mod,
        unicode="",
        scancode=0,
    )


def _click(x: int, y: int) -> pygame.event.Event:
    """Create a MOUSEBUTTONDOWN event."""
    return pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        button=1,
        pos=(x, y),
    )


# -------------------------------------------------------------------
# World context (no menu open)
# -------------------------------------------------------------------


class TestWorldKeys:
    """Key events in the world context produce valid results."""

    def test_move_up(self, ui, state_factory) -> None:
        """W key produces a movement or face action."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        result = ui.handle_event(_key(pygame.K_w), state)
        assert result.action is not None

    def test_mine(self, ui, state_factory) -> None:
        """Space produces MINE."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        result = ui.handle_event(_key(pygame.K_SPACE), state)
        assert result.action == int(Action.MINE)

    def test_interact_places_on_empty_tile(
        self,
        ui,
        state_factory,
    ) -> None:
        """E places selected machine when facing empty tile."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.selected_item = int(ItemType.MINER)
        result = ui.handle_event(_key(pygame.K_e), state)
        assert result.action == int(Action.PLACE_MINER)

    def test_interact_picks_up_machine(
        self,
        ui,
        state_factory,
    ) -> None:
        """E picks up when facing a machine."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.zeros(
                (8, 8),
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(int(Machine.MINER)),
        )
        result = ui.handle_event(_key(pygame.K_e), state)
        assert result.action == int(Action.PICKUP)

    def test_rotate_on_machine(self, ui, state_factory) -> None:
        """R facing a machine produces a ROTATE_* action."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.zeros(
                (8, 8),
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(int(Machine.MINER)),
            machine_direction=jnp.zeros(
                (8, 8),
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(int(Direction.DOWN)),
        )
        result = ui.handle_event(_key(pygame.K_r), state)
        # DOWN -> clockwise -> LEFT = ROTATE_LEFT
        assert result.action == int(Action.ROTATE_LEFT)

    def test_rotate_on_empty_is_noop(self, ui, state_factory) -> None:
        """R facing empty tile produces NOOP."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        result = ui.handle_event(_key(pygame.K_r), state)
        assert result.action == int(Action.NOOP)

    def test_slot_1(self, ui, state_factory) -> None:
        """1 key selects miner slot."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.handle_event(_key(pygame.K_1), state)
        assert ui.play_state.selected_item == int(ItemType.MINER)

    def test_toggle_hotbar_q(self, ui, state_factory) -> None:
        """Q key in world context does not crash (no-op stub)."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        result = ui.handle_event(_key(pygame.K_q), state)
        assert isinstance(result, GameUIResult)

    def test_player_select(self, ui, state_factory) -> None:
        """Ctrl+2 does not crash (requires params.num_players >= 2)."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            num_players=2,
        )
        result = ui.handle_event(
            _key(pygame.K_2, pygame.KMOD_CTRL),
            state,
        )
        assert isinstance(result, GameUIResult)


# -------------------------------------------------------------------
# Escape / pause
# -------------------------------------------------------------------


class TestEscapeAndPause:
    """Escape opens pause, navigates pause, and closes menus."""

    def test_escape_opens_pause(self, ui, state_factory) -> None:
        """Escape in world opens pause menu."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.handle_event(_key(pygame.K_ESCAPE), state)
        assert ui.play_state.pause_open is True

    def test_escape_closes_pause(self, ui, state_factory) -> None:
        """Escape in pause menu closes it."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.pause_open = True
        ui.handle_event(_key(pygame.K_ESCAPE), state)
        assert ui.play_state.pause_open is False

    def test_pause_nav_and_confirm(self, ui, state_factory) -> None:
        """Navigate pause menu and confirm resume."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.pause_open = True
        ui.play_state.pause_selection = 0
        ui.handle_event(_key(pygame.K_e), state)
        assert ui.play_state.pause_open is False

    def test_escape_closes_inventory(self, ui, state_factory) -> None:
        """Escape closes inventory."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.inventory_open = True
        ui.handle_event(_key(pygame.K_ESCAPE), state)
        assert ui.play_state.inventory_open is False

    def test_escape_closes_achievements(self, ui, state_factory) -> None:
        """Escape closes achievements."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.achievement_open = True
        ui.handle_event(_key(pygame.K_ESCAPE), state)
        assert ui.play_state.achievement_open is False

    def test_escape_closes_machine(self, ui, state_factory) -> None:
        """Escape closes machine menu."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.machine_open = True
        ui.handle_event(_key(pygame.K_ESCAPE), state)
        assert ui.play_state.machine_open is False


# -------------------------------------------------------------------
# Inventory context
# -------------------------------------------------------------------


class TestInventoryContext:
    """Keys in inventory/crafting panels."""

    def test_open_inventory(self, ui, state_factory) -> None:
        """I key opens inventory."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.handle_event(_key(pygame.K_i), state)
        assert ui.play_state.inventory_open is True

    def test_inventory_nav(self, ui, state_factory) -> None:
        """Arrow keys in inventory panel navigate."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ps = ui.play_state
        ps.inventory_open = True
        ps.menu_focus = "inventory"
        ps.selected_item = int(ItemType.MINER)
        result = ui.handle_event(_key(pygame.K_d), state)
        assert isinstance(result, GameUIResult)

    def test_crafting_confirm(self, ui, state_factory) -> None:
        """E in crafting panel emits a craft action."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ps = ui.play_state
        ps.inventory_open = True
        ps.menu_focus = "crafting"
        ps.selected_recipe = 0
        result = ui.handle_event(_key(pygame.K_e), state)
        assert result.action is not None


# -------------------------------------------------------------------
# Machine inspection context
# -------------------------------------------------------------------


class TestMachineContext:
    """Keys in machine inspection menu with a real machine on the map."""

    @pytest.fixture
    def machine_state(self, state_factory):
        """State with a miner at (1,1) and player facing it."""
        return state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.zeros(
                (8, 8),
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(int(Machine.MINER)),
        )

    def test_open_machine(self, ui, machine_state) -> None:
        """F key opens machine menu when facing a machine."""
        ui.handle_event(_key(pygame.K_f), machine_state)
        assert ui.play_state.machine_open is True

    def test_machine_nav_up_down(self, ui, machine_state) -> None:
        """W/S toggles machine panel focus."""
        ps = ui.play_state
        ps.machine_open = True
        ps.machine_tx, ps.machine_ty = 1, 1
        ps.machine_panel_active = True
        ui.handle_event(_key(pygame.K_w), machine_state)
        assert ps.machine_panel_active is False

    def test_machine_nav_left_right_player_panel(
        self,
        ui,
        machine_state,
    ) -> None:
        """A/D on player panel cycles selected item."""
        ps = ui.play_state
        ps.machine_open = True
        ps.machine_tx, ps.machine_ty = 1, 1
        ps.machine_panel_active = False
        ps.selected_item = 1
        ui.handle_event(_key(pygame.K_d), machine_state)
        assert ps.selected_item != 1

    def test_machine_nav_left_right_machine_panel(
        self,
        ui,
        machine_state,
    ) -> None:
        """A/D on machine panel cycles focused item (no crash)."""
        ps = ui.play_state
        ps.machine_open = True
        ps.machine_tx, ps.machine_ty = 1, 1
        ps.machine_panel_active = True
        ps.focused_machine_item = 0
        result = ui.handle_event(_key(pygame.K_d), machine_state)
        assert isinstance(result, GameUIResult)

    def test_machine_confirm_withdraw(self, ui, machine_state) -> None:
        """E on machine panel attempts withdraw."""
        ps = ui.play_state
        ps.machine_open = True
        ps.machine_tx, ps.machine_ty = 1, 1
        ps.machine_panel_active = True
        ps.focused_machine_item = int(ItemType.COAL)
        result = ui.handle_event(_key(pygame.K_e), machine_state)
        assert result.action is not None

    def test_machine_confirm_deposit(self, ui, machine_state) -> None:
        """E on player panel attempts deposit."""
        ps = ui.play_state
        ps.machine_open = True
        ps.machine_tx, ps.machine_ty = 1, 1
        ps.machine_panel_active = False
        ps.selected_item = int(ItemType.COAL)
        result = ui.handle_event(_key(pygame.K_e), machine_state)
        assert result.action is not None

    def test_machine_cycle_recipe(self, ui, machine_state) -> None:
        """Q in machine menu does not crash (recipe cycling removed)."""
        ps = ui.play_state
        ps.machine_open = True
        ps.machine_tx, ps.machine_ty = 1, 1
        result = ui.handle_event(_key(pygame.K_q), machine_state)
        assert isinstance(result, GameUIResult)


# -------------------------------------------------------------------
# Research context
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Achievement context
# -------------------------------------------------------------------


class TestAchievementContext:
    """Keys in the achievement menu."""

    def test_open_achievements(self, ui, state_factory) -> None:
        """P key opens achievements."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.handle_event(_key(pygame.K_p), state)
        assert ui.play_state.achievement_open is True

    def test_achievement_nav(self, ui, state_factory) -> None:
        """S key in achievements navigates down."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.achievement_open = True
        ui.play_state.achievement_selection = 0
        ui.handle_event(_key(pygame.K_s), state)
        assert ui.play_state.achievement_selection == 1


# -------------------------------------------------------------------
# Help context
# -------------------------------------------------------------------


class TestHelpContext:
    """Help overlay interaction."""

    def test_controls_from_pause_opens_help(
        self,
        ui,
        state_factory,
    ) -> None:
        """Selecting Controls in pause menu opens help overlay."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.pause_open = True
        ui.play_state.pause_selection = 1  # Controls
        ui.handle_event(_key(pygame.K_e), state)
        assert ui.play_state.help_open is True
        assert ui.play_state.pause_open is False

    def test_any_key_closes_help(self, ui, state_factory) -> None:
        """Any key closes the help overlay."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.help_open = True
        ui.handle_event(_key(pygame.K_a), state)
        assert ui.play_state.help_open is False


# -------------------------------------------------------------------
# Mouse click context
# -------------------------------------------------------------------


class TestMouseClick:
    """Mouse clicks on the world and UI buttons."""

    def test_click_world_no_placement(self, ui, state_factory) -> None:
        """Mouse click on world no longer places (use E key instead)."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.selected_item = int(ItemType.MINER)
        result = ui.handle_event(_click(100, 100), state)
        assert result.action is None


# -------------------------------------------------------------------
# Render frame (exercises render_frame with real state)
# -------------------------------------------------------------------


class TestRenderFrame:
    """render_frame does not crash with real state."""

    def test_basic_render(self, ui, state_factory) -> None:
        """Basic render produces correct shape."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        frame, regions = ui.render_frame(
            state,
            256,
            256,
            32,
            0,
            0,
        )
        assert frame.shape == (256, 256, 3)
        assert isinstance(regions, list)

    def test_render_with_machine_menu(self, ui, state_factory) -> None:
        """Render with machine menu open."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
            machine_types=jnp.zeros(
                (8, 8),
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(int(Machine.MINER)),
        )
        ps = ui.play_state
        ps.machine_open = True
        ps.machine_tx, ps.machine_ty = 1, 1
        frame, regions = ui.render_frame(
            state,
            256,
            256,
            32,
            0,
            0,
        )
        assert frame.shape == (256, 256, 3)

    def test_render_with_inventory(self, ui, state_factory) -> None:
        """Render with inventory open (needs larger canvas)."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.inventory_open = True
        frame, regions = ui.render_frame(
            state,
            512,
            512,
            32,
            0,
            0,
        )
        assert frame.shape == (512, 512, 3)

    def test_render_with_pause(self, ui, state_factory) -> None:
        """Render with pause menu open."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.pause_open = True
        frame, regions = ui.render_frame(
            state,
            256,
            256,
            32,
            0,
            0,
        )
        assert frame.shape == (256, 256, 3)

    def test_render_with_help(self, ui, state_factory) -> None:
        """Render with help overlay."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.play_state.help_open = True
        frame, regions = ui.render_frame(
            state,
            256,
            256,
            32,
            0,
            0,
        )
        assert frame.shape == (256, 256, 3)

    def test_update_hover(self, ui, state_factory) -> None:
        """update_hover does not crash."""
        state = state_factory(
            world_map=jnp.zeros((8, 8), dtype=jnp.int32),
        )
        ui.update_hover(state)
        assert ui.play_state.hover_tile_x >= -1
