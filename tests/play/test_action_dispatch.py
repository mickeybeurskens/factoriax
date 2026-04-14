"""Light integration tests for action dispatch through GameUI.

These tests exercise the full key-event-to-action path, catching refactor
drift between the Action enum and the UI layer. Each test creates a real
GameUI, simulates a pygame event, and verifies the emitted action integer
is valid.
"""

from __future__ import annotations

import jax.numpy as jnp
import pygame
import pytest

from factoriax.config import build_key_lookup, default_keyboard
from factoriax.constants import (
    CRAFT_BASE,
    DEPOSIT_BASE,
    MACHINE_INVENTORY_COUNT_DTYPE,
    NUM_ITEM_TYPES,
    Action,
    ItemType,
    MachineType,
)
from factoriax.play.game_ui import GameUI
from factoriax.state import EnvParams


@pytest.fixture
def params() -> EnvParams:
    """Return small environment parameters."""
    return EnvParams(map_width=4, map_height=4, num_players=1)


@pytest.fixture
def game_ui(params: EnvParams) -> GameUI:
    """Create a GameUI with default bindings and no welcome screen."""
    kb = build_key_lookup(default_keyboard())
    return GameUI(params, kb, welcome_open=False)


def _confirm_key() -> int:
    """Return the pygame key code for the CONFIRM action (Enter)."""
    return pygame.K_RETURN


def _make_keydown(key: int) -> pygame.event.Event:
    """Create a KEYDOWN event for the given key."""
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)


# ---------------------------------------------------------------------------
# Deposit
# ---------------------------------------------------------------------------


class TestDepositAction:
    """CONFIRM in machine menu with player panel active emits DEPOSIT_*."""

    def test_deposit_emits_valid_action(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Depositing coal emits DEPOSIT_COAL."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.PALLET), dtype=jnp.int32,
            ),
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = False  # player panel active
        ps.selected_item = int(ItemType.COAL)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        expected = DEPOSIT_BASE + int(ItemType.COAL) - int(ItemType.COAL)
        assert result.action == expected
        assert result.action == int(Action.DEPOSIT_COAL)

    def test_deposit_iron(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Depositing iron emits DEPOSIT_IRON."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.PALLET), dtype=jnp.int32,
            ),
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = False
        ps.selected_item = int(ItemType.IRON_ORE)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.DEPOSIT_IRON_ORE)

    def test_deposit_empty_type_is_noop(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Depositing with EMPTY selected emits no action."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.PALLET), dtype=jnp.int32,
            ),
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = False
        ps.selected_item = int(ItemType.EMPTY)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action is None


# ---------------------------------------------------------------------------
# Withdraw
# ---------------------------------------------------------------------------


class TestWithdrawAction:
    """CONFIRM in machine menu with machine panel active emits WITHDRAW_*."""

    def test_withdraw_emits_valid_action(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Withdrawing coal emits WITHDRAW_COAL."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[0, 0, int(ItemType.COAL)].set(10)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = True  # machine panel active
        ps.machine_tx, ps.machine_ty = 0, 0
        ps.focused_machine_item = int(ItemType.COAL)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.WITHDRAW_COAL)

    def test_withdraw_copper(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Withdrawing copper emits WITHDRAW_COPPER."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[1, 1, int(ItemType.COPPER_ORE)].set(5)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = True
        ps.machine_tx, ps.machine_ty = 1, 1
        ps.focused_machine_item = int(ItemType.COPPER_ORE)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.WITHDRAW_COPPER_ORE)


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


class TestResearchAction:
    """CONFIRM in research menu emits RESEARCH_BASIC + selection offset."""

    def test_research_basic(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """First research option emits RESEARCH_BASIC."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.research_open = True
        ps.research_selection = 0

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.RESEARCH_BASIC)

    def test_research_advanced(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Second research option emits RESEARCH_ADVANCED."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.research_open = True
        ps.research_selection = 1

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.RESEARCH_ADVANCED)


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------


class TestCraftAction:
    """CONFIRM in crafting panel emits CRAFT_BASE + recipe offset."""

    def test_craft_first_recipe(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """First recipe (Iron Plate) emits CRAFT_IRON_PLATE."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.inventory_open = True
        ps.menu_focus = "crafting"
        ps.selected_recipe = 0

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.CRAFT_IRON_PLATE)

    def test_craft_second_recipe(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Second recipe (Copper Plate) emits CRAFT_COPPER_PLATE."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.inventory_open = True
        ps.menu_focus = "crafting"
        ps.selected_recipe = 1

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.CRAFT_COPPER_PLATE)

    def test_craft_action_uses_craft_base(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """All craft actions are CRAFT_BASE + recipe index."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.inventory_open = True
        ps.menu_focus = "crafting"

        for recipe_idx in range(3):
            ps.selected_recipe = recipe_idx
            result = game_ui.handle_event(
                _make_keydown(_confirm_key()), state,
            )
            assert result.action == CRAFT_BASE + recipe_idx
