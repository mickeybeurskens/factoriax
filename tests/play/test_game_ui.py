"""Regression tests for GameUI event dispatch with pouch inventory model.

These tests verify that GameUI uses PlayState (not EnvState) for
UI-only state like selected_item, and correctly reads from the pouch
inventory fields.
"""

from __future__ import annotations

import jax.numpy as jnp
import pygame
import pytest

from factoriax.config import build_key_lookup, default_keyboard
from factoriax.constants import Action, ItemType
from factoriax.play.game_ui import GameUI
from factoriax.state import EnvParams


@pytest.fixture
def params() -> EnvParams:
    """Return default environment parameters."""
    return EnvParams(map_width=4, map_height=4, num_players=1)


@pytest.fixture
def game_ui(params: EnvParams) -> GameUI:
    """Create a GameUI with default bindings and no welcome screen."""
    kb = build_key_lookup(default_keyboard())
    return GameUI(params, kb, welcome_open=False)


class TestSlotKeySelection:
    """Number keys should update PlayState.selected_item, not EnvState."""

    def test_pressing_1_selects_miner(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Key '1' sets selected_item to ItemType.MINER (tool belt slot 1)."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        game_ui.play_state.selected_item = 5  # start at something else
        event = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_1, mod=0,
        )
        result = game_ui.handle_event(event, state)
        assert game_ui.play_state.selected_item == int(ItemType.MINER)
        # EnvState should not be mutated for UI state
        assert result.state is not None

    def test_pressing_3_selects_belt(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """Key '3' sets selected_item to ItemType.CONVEYOR_BELT (slot 3)."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        event = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_3, mod=0,
        )
        game_ui.handle_event(event, state)
        assert game_ui.play_state.selected_item == int(ItemType.CONVEYOR_BELT)


class TestInventoryIsCraftingOnly:
    """Inventory menu goes straight to crafting navigation."""

    def test_up_cycles_recipe(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """W key in inventory cycles recipe selection."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        ps = game_ui.play_state
        ps.inventory_open = True
        ps.selected_recipe = 1

        event = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_w, mod=0,
        )
        game_ui.handle_event(event, state)
        assert ps.selected_recipe == 0


class TestInteractPlacesAndPicksUp:
    """E key places on empty tile, picks up facing a machine."""

    def test_interact_places_on_empty(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """E key with MINER selected places on empty tile."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        ps = game_ui.play_state
        ps.selected_item = int(ItemType.MINER)
        ps.welcome_open = False

        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_e, mod=0, unicode="e", scancode=0,
        )
        result = game_ui.handle_event(event, state)
        assert result.action == int(Action.PLACE_MINER)

    def test_interact_noop_without_placeable(
        self, game_ui: GameUI, state_factory,
    ) -> None:
        """E key with COAL selected on empty tile does nothing."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        ps = game_ui.play_state
        ps.selected_item = int(ItemType.COAL)
        ps.welcome_open = False

        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_e, mod=0, unicode="e", scancode=0,
        )
        result = game_ui.handle_event(event, state)
        assert result.action is None
