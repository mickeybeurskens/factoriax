"""Tests for controller binding resolution and input name formatting."""

from __future__ import annotations

from unittest.mock import MagicMock

import pygame
import pytest

from factoriax.config import (
    _AXIS_DEADZONE,
    build_controller_lookup,
    controller_event_to_name,
    default_controller,
    event_to_key_name,
    resolve_controller_axis,
    resolve_controller_button,
    resolve_controller_hat,
)

# ---------------------------------------------------------------------------
# build_controller_lookup
# ---------------------------------------------------------------------------


class TestBuildControllerLookup:
    """Tests for inverting controller bindings into a lookup dict."""

    def test_single_binding(self) -> None:
        """One action mapped to one input produces a correct entry."""
        bindings = {"mine": ["BUTTON_2"]}
        lookup = build_controller_lookup(bindings)
        assert lookup["BUTTON_2"] == frozenset({"mine"})

    def test_multiple_actions_same_input(self) -> None:
        """Two actions on the same input appear in the frozenset."""
        bindings = {"confirm": ["BUTTON_0"], "interact": ["BUTTON_0"]}
        lookup = build_controller_lookup(bindings)
        assert lookup["BUTTON_0"] == frozenset({"confirm", "interact"})

    def test_multiple_inputs_same_action(self) -> None:
        """One action with two inputs creates two entries."""
        bindings = {"move_up": ["AXIS_1_NEG", "HAT_0_UP"]}
        lookup = build_controller_lookup(bindings)
        assert "move_up" in lookup["AXIS_1_NEG"]
        assert "move_up" in lookup["HAT_0_UP"]

    def test_empty_bindings(self) -> None:
        """Empty bindings produce an empty lookup."""
        assert build_controller_lookup({}) == {}

    def test_default_controller_builds(self) -> None:
        """Default controller bindings build without error."""
        lookup = build_controller_lookup(default_controller())
        assert len(lookup) > 0


# ---------------------------------------------------------------------------
# resolve_controller_button
# ---------------------------------------------------------------------------


class TestResolveControllerButton:
    """Tests for button index resolution."""

    def test_bound_button(self) -> None:
        """A bound button returns its actions."""
        lookup = build_controller_lookup({"mine": ["BUTTON_2"]})
        assert resolve_controller_button(lookup, 2) == frozenset({"mine"})

    def test_unbound_button(self) -> None:
        """An unbound button returns an empty frozenset."""
        lookup = build_controller_lookup({"mine": ["BUTTON_2"]})
        assert resolve_controller_button(lookup, 9) == frozenset()


# ---------------------------------------------------------------------------
# resolve_controller_hat
# ---------------------------------------------------------------------------


class TestResolveControllerHat:
    """Tests for hat/d-pad resolution."""

    @pytest.fixture
    def lookup(self) -> dict[str, frozenset[str]]:
        """Build a lookup with d-pad navigation bindings."""
        return build_controller_lookup({
            "nav_up": ["HAT_0_UP"],
            "nav_down": ["HAT_0_DOWN"],
            "nav_left": ["HAT_0_LEFT"],
            "nav_right": ["HAT_0_RIGHT"],
        })

    def test_up(self, lookup) -> None:
        """Hat up returns nav_up."""
        assert resolve_controller_hat(lookup, 0, (0, 1)) == frozenset(
            {"nav_up"},
        )

    def test_down(self, lookup) -> None:
        """Hat down returns nav_down."""
        assert resolve_controller_hat(lookup, 0, (0, -1)) == frozenset(
            {"nav_down"},
        )

    def test_left(self, lookup) -> None:
        """Hat left returns nav_left."""
        assert resolve_controller_hat(lookup, 0, (-1, 0)) == frozenset(
            {"nav_left"},
        )

    def test_right(self, lookup) -> None:
        """Hat right returns nav_right."""
        assert resolve_controller_hat(lookup, 0, (1, 0)) == frozenset(
            {"nav_right"},
        )

    def test_center(self, lookup) -> None:
        """Hat center returns empty."""
        assert resolve_controller_hat(lookup, 0, (0, 0)) == frozenset()

    def test_diagonal(self, lookup) -> None:
        """Diagonal hat returns both directions."""
        result = resolve_controller_hat(lookup, 0, (1, 1))
        assert result == frozenset({"nav_right", "nav_up"})


# ---------------------------------------------------------------------------
# resolve_controller_axis
# ---------------------------------------------------------------------------


class TestResolveControllerAxis:
    """Tests for axis deadzone resolution."""

    @pytest.fixture
    def lookup(self) -> dict[str, frozenset[str]]:
        """Build a lookup with stick movement bindings."""
        return build_controller_lookup({
            "move_up": ["AXIS_1_NEG"],
            "move_down": ["AXIS_1_POS"],
        })

    def test_positive_past_deadzone(self, lookup) -> None:
        """Value above deadzone resolves to POS actions."""
        result = resolve_controller_axis(lookup, 1, _AXIS_DEADZONE + 0.1)
        assert result == frozenset({"move_down"})

    def test_negative_past_deadzone(self, lookup) -> None:
        """Value below negative deadzone resolves to NEG actions."""
        result = resolve_controller_axis(lookup, 1, -_AXIS_DEADZONE - 0.1)
        assert result == frozenset({"move_up"})

    def test_within_deadzone(self, lookup) -> None:
        """Value within deadzone returns empty."""
        assert resolve_controller_axis(lookup, 1, 0.1) == frozenset()
        assert resolve_controller_axis(lookup, 1, -0.1) == frozenset()
        assert resolve_controller_axis(lookup, 1, 0.0) == frozenset()

    def test_exactly_at_deadzone(self, lookup) -> None:
        """Value exactly at the deadzone boundary returns empty."""
        assert resolve_controller_axis(
            lookup, 1, _AXIS_DEADZONE,
        ) == frozenset()
        assert resolve_controller_axis(
            lookup, 1, -_AXIS_DEADZONE,
        ) == frozenset()


# ---------------------------------------------------------------------------
# event_to_key_name
# ---------------------------------------------------------------------------


class TestEventToKeyName:
    """Tests for formatting key events as binding name strings."""

    def test_bare_letter(self) -> None:
        """A plain letter key formats as K_<letter>."""
        result = event_to_key_name(pygame.K_w, 0)
        assert result == "K_w"

    def test_special_key(self) -> None:
        """A special key uses the pygame constant name."""
        result = event_to_key_name(pygame.K_SPACE, 0)
        assert result == "K_SPACE"

    def test_shift_modifier(self) -> None:
        """Shift modifier prepends SHIFT+."""
        result = event_to_key_name(pygame.K_1, pygame.KMOD_SHIFT)
        assert result == "SHIFT+K_1"

    def test_ctrl_modifier(self) -> None:
        """Ctrl modifier prepends CTRL+."""
        result = event_to_key_name(pygame.K_1, pygame.KMOD_CTRL)
        assert result == "CTRL+K_1"

    def test_ctrl_shift_modifier(self) -> None:
        """Both modifiers produce CTRL+SHIFT+ prefix."""
        mods = pygame.KMOD_CTRL | pygame.KMOD_SHIFT
        result = event_to_key_name(pygame.K_1, mods)
        assert result == "CTRL+SHIFT+K_1"


# ---------------------------------------------------------------------------
# controller_event_to_name
# ---------------------------------------------------------------------------


class TestControllerEventToName:
    """Tests for formatting controller events as binding name strings."""

    def test_button(self) -> None:
        """JOYBUTTONDOWN formats as BUTTON_n."""
        event = MagicMock()
        event.type = pygame.JOYBUTTONDOWN
        event.button = 3
        assert controller_event_to_name(event) == "BUTTON_3"

    def test_hat_up(self) -> None:
        """JOYHATMOTION up formats as HAT_n_UP."""
        event = MagicMock()
        event.type = pygame.JOYHATMOTION
        event.hat = 0
        event.value = (0, 1)
        assert controller_event_to_name(event) == "HAT_0_UP"

    def test_hat_center(self) -> None:
        """JOYHATMOTION center returns None."""
        event = MagicMock()
        event.type = pygame.JOYHATMOTION
        event.hat = 0
        event.value = (0, 0)
        assert controller_event_to_name(event) is None

    def test_axis_positive(self) -> None:
        """JOYAXISMOTION past positive deadzone formats as AXIS_n_POS."""
        event = MagicMock()
        event.type = pygame.JOYAXISMOTION
        event.axis = 1
        event.value = 0.8
        assert controller_event_to_name(event) == "AXIS_1_POS"

    def test_axis_negative(self) -> None:
        """JOYAXISMOTION past negative deadzone formats as AXIS_n_NEG."""
        event = MagicMock()
        event.type = pygame.JOYAXISMOTION
        event.axis = 0
        event.value = -0.9
        assert controller_event_to_name(event) == "AXIS_0_NEG"

    def test_axis_within_deadzone(self) -> None:
        """JOYAXISMOTION within deadzone returns None."""
        event = MagicMock()
        event.type = pygame.JOYAXISMOTION
        event.axis = 0
        event.value = 0.1
        assert controller_event_to_name(event) is None

    def test_unknown_event(self) -> None:
        """Unrecognized event type returns None."""
        event = MagicMock()
        event.type = pygame.MOUSEMOTION
        assert controller_event_to_name(event) is None
