"""Tests for :mod:`factoriax.playground.menu.controls_menu`.

The controls menu shows a binding and lets a player rebind it. Two parts of
that carry a contract. ``_format_binding`` turns a stored binding name into
the text on screen, and ``_init_rebind_actions`` decides which actions the
menu offers to rebind.
"""

from __future__ import annotations

from factoriax.playground.config import (
    PlayerAction,
    default_controller,
    default_keyboard,
)
from factoriax.playground.menu.controls_menu import (
    _format_binding,
    _format_controller_display,
    _format_key_display,
    _init_rebind_actions,
)


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
                assert action_key in kb, f"{action_key} missing from default_keyboard"

    def test_actions_exist_in_default_controller(self) -> None:
        """Every listed action has a default controller binding entry."""
        ctrl = default_controller()
        groups = _init_rebind_actions()
        for _, actions in groups:
            for action_key, _ in actions:
                assert action_key in ctrl, (
                    f"{action_key} missing from default_controller"
                )
