"""Integration tests for the rebinding system.

Exercises the full round-trip: config bindings -> lookup build ->
resolution -> dispatch, and verifies that mutating bindings in a
PlayerConfig produces the expected behavior change.
"""

from __future__ import annotations

import jax.numpy as jnp
import pygame
import pytest

from factoriax.config import (
    PlayerAction,
    PlayerConfig,
    build_controller_lookup,
    build_key_lookup,
    default_controller,
    default_keyboard,
    resolve_controller_button,
    resolve_controller_hat,
    resolve_key,
)
from factoriax.menu.settings_menu import (
    _format_binding,
    _format_controller_display,
    _format_key_display,
    _init_rebind_actions,
)
from factoriax.play.game_ui import GameUI
from factoriax.state import EnvParams

# ---------------------------------------------------------------------------
# Binding mutation round-trip
# ---------------------------------------------------------------------------


class TestKeyboardRebindRoundTrip:
    """Rebinding a keyboard key changes what resolve_key returns."""

    def test_rebind_mine_to_new_key(self) -> None:
        """Changing mine from SPACE to K_x makes K_x trigger mine."""
        bindings = default_keyboard()
        assert "K_SPACE" in bindings[PlayerAction.MINE]

        # Rebind mine to X.
        bindings[PlayerAction.MINE] = ["K_x"]

        lookup = build_key_lookup(bindings)
        actions = resolve_key(lookup, pygame.K_x)
        assert PlayerAction.MINE in actions

        # Old key no longer triggers mine.
        old_actions = resolve_key(lookup, pygame.K_SPACE)
        assert PlayerAction.MINE not in old_actions

    def test_rebind_preserves_other_actions(self) -> None:
        """Rebinding one action does not affect unrelated actions."""
        bindings = default_keyboard()
        bindings[PlayerAction.MINE] = ["K_x"]

        lookup = build_key_lookup(bindings)
        move_actions = resolve_key(lookup, pygame.K_w)
        assert PlayerAction.MOVE_UP in move_actions

    def test_rebind_with_modifier(self) -> None:
        """Rebinding to a modified key works in resolution."""
        bindings = default_keyboard()
        bindings[PlayerAction.MINE] = ["SHIFT+K_m"]

        lookup = build_key_lookup(bindings)
        actions = resolve_key(lookup, pygame.K_m, pygame.KMOD_SHIFT)
        assert PlayerAction.MINE in actions

        bare = resolve_key(lookup, pygame.K_m)
        assert PlayerAction.MINE not in bare


class TestControllerRebindRoundTrip:
    """Rebinding a controller input changes resolution."""

    def test_rebind_button(self) -> None:
        """Changing mine from BUTTON_2 to BUTTON_9 takes effect."""
        bindings = default_controller()
        bindings[PlayerAction.MINE] = ["BUTTON_9"]

        lookup = build_controller_lookup(bindings)
        actions = resolve_controller_button(lookup, 9)
        assert PlayerAction.MINE in actions

        old = resolve_controller_button(lookup, 2)
        assert PlayerAction.MINE not in old

    def test_rebind_hat(self) -> None:
        """Rebinding nav_up from HAT_0_UP to BUTTON_4 removes hat."""
        bindings = default_controller()
        bindings[PlayerAction.NAV_UP] = ["BUTTON_4"]

        lookup = build_controller_lookup(bindings)
        hat_actions = resolve_controller_hat(lookup, 0, (0, 1))
        assert PlayerAction.NAV_UP not in hat_actions

        btn_actions = resolve_controller_button(lookup, 4)
        assert PlayerAction.NAV_UP in btn_actions


# ---------------------------------------------------------------------------
# Config persistence simulation
# ---------------------------------------------------------------------------


class TestConfigBindingMutation:
    """Simulates the rebinding flow: mutate config, rebuild lookups."""

    def test_mutate_keyboard_in_config(self) -> None:
        """Mutating config.keyboard in place affects new lookups."""
        config = PlayerConfig(
            keyboard=default_keyboard(),
            controller=default_controller(),
        )
        config.keyboard[PlayerAction.INTERACT] = ["K_j"]

        lookup = build_key_lookup(config.keyboard)
        actions = resolve_key(lookup, pygame.K_j)
        assert PlayerAction.INTERACT in actions

    def test_mutate_controller_in_config(self) -> None:
        """Mutating config.controller in place affects new lookups."""
        config = PlayerConfig(
            keyboard=default_keyboard(),
            controller=default_controller(),
        )
        config.controller[PlayerAction.INTERACT] = ["BUTTON_7"]

        lookup = build_controller_lookup(config.controller)
        actions = resolve_controller_button(lookup, 7)
        assert PlayerAction.INTERACT in actions

    def test_reset_to_defaults(self) -> None:
        """Replacing bindings with defaults restores original mapping."""
        config = PlayerConfig(
            keyboard=default_keyboard(),
            controller=default_controller(),
        )
        config.keyboard[PlayerAction.MINE] = ["K_z"]

        # Reset.
        config.keyboard = default_keyboard()
        lookup = build_key_lookup(config.keyboard)

        z_actions = resolve_key(lookup, pygame.K_z)
        assert PlayerAction.MINE not in z_actions

        space_actions = resolve_key(lookup, pygame.K_SPACE)
        assert PlayerAction.MINE in space_actions


# ---------------------------------------------------------------------------
# GameUI integration: rebound key dispatches correctly
# ---------------------------------------------------------------------------


class TestGameUIWithReboundKey:
    """A rebound key produces the correct action through GameUI."""

    @pytest.fixture
    def params(self) -> EnvParams:
        """Return small environment parameters."""
        return EnvParams(map_width=4, map_height=4, num_players=1)

    def test_rebound_mine_key_dispatches(
        self, params, state_factory,
    ) -> None:
        """Rebinding mine to K_j makes J produce MINE action."""
        from factoriax.constants import Action

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
            key=pygame.K_j, mod=0, unicode="j", scancode=0,
        )
        result = ui.handle_event(event, state)
        assert result.action == int(Action.MINE)

    def test_old_key_no_longer_dispatches(
        self, params, state_factory,
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
            key=pygame.K_SPACE, mod=0, unicode=" ", scancode=0,
        )
        result = ui.handle_event(event, state)
        assert result.action is None


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


class TestBindingFormatting:
    """The formatting helpers produce readable strings."""

    def test_keyboard_single_key(self) -> None:
        """Single letter key formats without prefix."""
        assert _format_key_display(["K_w"]) == "w"

    def test_keyboard_multiple_keys(self) -> None:
        """Multiple keys joined with comma."""
        assert _format_key_display(["K_w", "K_UP"]) == "w, UP"

    def test_keyboard_modifier(self) -> None:
        """Modifier formats as Capitalized prefix."""
        assert _format_key_display(["SHIFT+K_1"]) == "Shift+1"

    def test_keyboard_empty(self) -> None:
        """Empty list shows dash."""
        assert _format_key_display([]) == "-"

    def test_controller_button(self) -> None:
        """Controller button uses friendly name."""
        assert _format_controller_display(["BUTTON_0"]) == "A / Cross"

    def test_controller_axis(self) -> None:
        """Controller axis uses friendly name."""
        result = _format_controller_display(["AXIS_1_NEG"])
        assert result == "L-Stick Up"

    def test_controller_hat(self) -> None:
        """Controller hat uses friendly name."""
        result = _format_controller_display(["HAT_0_LEFT"])
        assert result == "D-pad Left"

    def test_controller_unknown(self) -> None:
        """Unknown controller input shows raw name."""
        assert _format_controller_display(["AXIS_5_POS"]) == "AXIS_5_POS"

    def test_controller_empty(self) -> None:
        """Empty list shows dash."""
        assert _format_controller_display([]) == "-"

    def test_format_binding_delegates(self) -> None:
        """_format_binding delegates to the correct formatter."""
        assert _format_binding(["K_w"], "keyboard") == "w"
        assert _format_binding(["BUTTON_0"], "controller") == "A / Cross"


# ---------------------------------------------------------------------------
# Rebind action list
# ---------------------------------------------------------------------------


class TestRebindActions:
    """The grouped action list covers expected actions."""

    def test_all_categories_present(self) -> None:
        """All three categories are present."""
        groups = _init_rebind_actions()
        cats = [cat for cat, _ in groups]
        assert "Movement" in cats
        assert "Actions" in cats
        assert "Menus" in cats

    def test_actions_are_valid_player_actions(self) -> None:
        """Every action key in the list is a valid PlayerAction."""
        groups = _init_rebind_actions()
        valid = set(PlayerAction)
        for _, actions in groups:
            for action_key, _ in actions:
                assert action_key in valid, f"{action_key} not valid"

    def test_actions_exist_in_default_keyboard(self) -> None:
        """Every listed action has a default keyboard binding."""
        kb = default_keyboard()
        groups = _init_rebind_actions()
        for _, actions in groups:
            for action_key, _ in actions:
                assert action_key in kb, (
                    f"{action_key} missing from default_keyboard"
                )

    def test_actions_exist_in_default_controller(self) -> None:
        """Every listed action has a default controller binding entry."""
        ctrl = default_controller()
        groups = _init_rebind_actions()
        for _, actions in groups:
            for action_key, _ in actions:
                assert action_key in ctrl, (
                    f"{action_key} missing from default_controller"
                )
