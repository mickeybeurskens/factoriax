"""A rebound key reaches the engine as the right action.

This test spans three subpackages, so it is not in the mirror.
``playground/config.py`` holds the binding table, ``playground/play/game_ui.py``
turns a key event into an action, and the assertion names an
``engine.constants.Action``. A rebind that the lookup accepts but the UI
ignores passes every test that stays inside one of those three.

``GameUI.handle_event`` reads ``pygame.key.get_mods()``, so this file asks
for the display by name. There is no autouse display outside
``tests/playground/``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pygame
import pytest

from factoriax.engine.state import EnvParams
from factoriax.playground.config import (
    PlayerAction,
    build_key_lookup,
    default_keyboard,
)
from factoriax.playground.play.game_ui import GameUI


@pytest.fixture(autouse=True)
def _display(pygame_display: None) -> None:
    """Request the session display. ``handle_event`` reads the video system."""


class TestGameUIWithReboundKey:
    """A rebound key produces the correct action through GameUI."""

    @pytest.fixture
    def params(self) -> EnvParams:
        """Return small environment parameters."""
        return EnvParams()

    def test_rebound_mine_key_dispatches(
        self,
        params,
        state_factory,
    ) -> None:
        """Rebinding mine to K_j makes J produce MINE action."""
        from factoriax.engine.constants import Action

        bindings = default_keyboard()
        bindings[PlayerAction.MINE] = ["K_j"]
        lookup = build_key_lookup(bindings)

        ui = GameUI(params, lookup)
        ui.play_state.welcome_open = False

        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )

        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_j,
            mod=0,
            unicode="j",
            scancode=0,
        )
        result = ui.handle_event(event, state)
        assert result.action == int(Action.MINE)

    def test_old_key_no_longer_dispatches(
        self,
        params,
        state_factory,
    ) -> None:
        """After rebinding mine away from SPACE, SPACE does nothing."""
        bindings = default_keyboard()
        bindings[PlayerAction.MINE] = ["K_j"]
        lookup = build_key_lookup(bindings)

        ui = GameUI(params, lookup)
        ui.play_state.welcome_open = False

        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )

        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_SPACE,
            mod=0,
            unicode=" ",
            scancode=0,
        )
        result = ui.handle_event(event, state)
        assert result.action is None


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------
