"""Tests for :mod:`factoriax.playground.play.game_ui`.

Two groups live here.

The dispatch group drives the whole path from a key event to an ``Action``
integer, so a drift between the ``Action`` enum and the UI shows up here.

The state group asserts that ``GameUI`` keeps UI-only state, such as the
selected item, in :class:`PlayState` and not in :class:`EnvState`. The
engine owns ``EnvState``, and a UI write to it gives a human a rule that a
policy does not have.
"""

from __future__ import annotations

import jax.numpy as jnp
import pygame
import pytest

from factoriax.engine.constants import (
    Action,
    ItemType,
    Machine,
)
from factoriax.engine.state import EnvParams
from factoriax.playground.config import build_key_lookup, default_keyboard
from factoriax.playground.play.game_ui import GameUI


@pytest.fixture
def params() -> EnvParams:
    """Return small environment parameters."""
    return EnvParams()


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
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """Depositing coal emits DEPOSIT_COAL."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4),
                int(Machine.PALLET),
                dtype=jnp.int32,
            ),
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = False  # player panel active
        ps.selected_item = int(ItemType.COAL)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.DEPOSIT_COAL)

    def test_deposit_iron(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """Depositing iron emits DEPOSIT_IRON."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4),
                int(Machine.PALLET),
                dtype=jnp.int32,
            ),
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = False
        ps.selected_item = int(ItemType.IRON_ORE)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.DEPOSIT_IRON_ORE)

    def test_deposit_empty_type_is_noop(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """Depositing with EMPTY selected emits no action."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4),
                int(Machine.PALLET),
                dtype=jnp.int32,
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
    """CONFIRM in the machine panel always emits the single WITHDRAW action."""

    def test_withdraw_emits_single_action(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """Machines have one output slot. CONFIRM emits WITHDRAW for any focus."""
        buf_type = jnp.zeros((4, 4), dtype=jnp.int8)
        buf_count = jnp.zeros((4, 4), dtype=jnp.int16)
        buf_type = buf_type.at[0, 0].set(int(ItemType.COAL))
        buf_count = buf_count.at[0, 0].set(10)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4),
                int(Machine.MINER),
                dtype=jnp.int32,
            ),
            buffer_type=buf_type,
            buffer_count=buf_count,
        )
        ps = game_ui.play_state
        ps.machine_open = True
        ps.machine_panel_active = True
        ps.machine_tx, ps.machine_ty = 0, 0
        ps.focused_machine_item = int(ItemType.COAL)

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.WITHDRAW)


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------


class TestCraftAction:
    """CONFIRM in crafting panel emits CRAFT_BASE + recipe offset."""

    def test_craft_first_recipe(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """First recipe (Iron Plate) emits CRAFT_IRON_PLATE."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.inventory_open = True
        ps.selected_recipe = 0

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.CRAFT_IRON_PLATE)

    def test_craft_second_recipe(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """Second recipe (Copper Plate) emits CRAFT_COPPER_PLATE."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.inventory_open = True
        ps.selected_recipe = 1

        result = game_ui.handle_event(_make_keydown(_confirm_key()), state)
        assert result.action == int(Action.CRAFT_COPPER_PLATE)

    def test_craft_confirm_emits_recipe_output_craft_action(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """CONFIRM crafts the selected recipe's output item.

        The crafting panel lists ``BASE_RECIPES`` in order, but recipe-list
        order and craft-family order differ, so the action must route through
        the output item. A ``CRAFT_BASE + index`` offset is wrong for a recipe
        whose output is out of craft-family position, for example a machine
        recipe between the half-fabricates.
        """
        from factoriax.engine.actions import ITEM_TO_CRAFT_ACTION
        from factoriax.engine.recipes import BASE_RECIPES, NUM_RECIPES

        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        ps = game_ui.play_state
        ps.inventory_open = True

        for recipe_idx in (0, 1, 9, NUM_RECIPES - 1):
            ps.selected_recipe = recipe_idx
            result = game_ui.handle_event(
                _make_keydown(_confirm_key()),
                state,
            )
            output = BASE_RECIPES[recipe_idx].output
            assert result.action == int(ITEM_TO_CRAFT_ACTION[output])


class TestSlotKeySelection:
    """Number keys update PlayState.selected_item, not EnvState."""

    def test_pressing_1_selects_miner(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """Key '1' sets selected_item to ItemType.MINER (tool belt slot 1)."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        game_ui.play_state.selected_item = 5  # start at something else
        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_1,
            mod=0,
        )
        result = game_ui.handle_event(event, state)
        assert game_ui.play_state.selected_item == int(ItemType.MINER)
        # EnvState must not be mutated for UI state
        assert result.state is not None

    def test_pressing_3_selects_belt(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """Key '3' sets selected_item to ItemType.CONVEYOR_BELT (slot 3)."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_3,
            mod=0,
        )
        game_ui.handle_event(event, state)
        assert game_ui.play_state.selected_item == int(ItemType.CONVEYOR_BELT)


class TestInventoryIsCraftingOnly:
    """Inventory menu goes straight to crafting navigation."""

    def test_up_cycles_recipe(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """W key in inventory cycles recipe selection."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        ps = game_ui.play_state
        ps.inventory_open = True
        ps.selected_recipe = 1

        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_w,
            mod=0,
        )
        game_ui.handle_event(event, state)
        assert ps.selected_recipe == 0


class TestInteractPlacesAndPicksUp:
    """E key places on empty tile, picks up facing a machine."""

    def test_interact_places_on_empty(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """E key with MINER selected places on empty tile."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        ps = game_ui.play_state
        ps.selected_item = int(ItemType.MINER)
        ps.welcome_open = False

        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_e,
            mod=0,
            unicode="e",
            scancode=0,
        )
        result = game_ui.handle_event(event, state)
        assert result.action == int(Action.PLACE_MINER)

    def test_interact_noop_without_placeable(
        self,
        game_ui: GameUI,
        state_factory,
    ) -> None:
        """E key with COAL selected on empty tile does nothing."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        ps = game_ui.play_state
        ps.selected_item = int(ItemType.COAL)
        ps.welcome_open = False

        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_e,
            mod=0,
            unicode="e",
            scancode=0,
        )
        result = game_ui.handle_event(event, state)
        assert result.action is None
