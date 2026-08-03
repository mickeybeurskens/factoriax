"""Tests for the keyboard half of :mod:`factoriax.playground.config`.

``default_keyboard`` gives the shipped bindings. ``build_key_lookup`` turns a
binding table into the lookup that the play UI reads, and ``resolve_key``
answers which :class:`PlayerAction` a key event names. A rebind changes the
table, and the lookup must follow it.
"""

from __future__ import annotations

import pygame
import pytest

from factoriax.playground.config import (
    KeyLookup,
    PlayerAction,
    build_key_lookup,
    default_controller,
    default_keyboard,
    resolve_key,
)

# Ensure pygame constants are available for key lookups.
pygame.init()


class TestDefaultBindings:
    """Verify default binding maps are complete and consistent."""

    def test_keyboard_covers_all_actions(self) -> None:
        """Every PlayerAction appears in the default keyboard map."""
        kb = default_keyboard()
        for action in PlayerAction:
            assert action in kb, f"Missing keyboard binding for {action}"

    def test_controller_covers_gameplay_and_nav(self) -> None:
        """Controller defaults cover movement and core navigation."""
        ctrl = default_controller()
        essential = [
            PlayerAction.MOVE_UP,
            PlayerAction.MOVE_DOWN,
            PlayerAction.MOVE_LEFT,
            PlayerAction.MOVE_RIGHT,
            PlayerAction.MINE,
            PlayerAction.INTERACT,
            PlayerAction.CONFIRM,
            PlayerAction.BACK,
            PlayerAction.NAV_UP,
            PlayerAction.NAV_DOWN,
        ]
        for action in essential:
            assert action in ctrl, f"Missing controller binding for {action}"
            assert len(ctrl[action]) > 0, f"Empty controller binding for {action}"

    def test_keyboard_movement_has_wasd_and_arrows(self) -> None:
        """Movement binds to both WASD and the arrow keys."""
        kb = default_keyboard()
        assert "K_w" in kb[PlayerAction.MOVE_UP]
        assert "K_UP" in kb[PlayerAction.MOVE_UP]
        assert "K_s" in kb[PlayerAction.MOVE_DOWN]
        assert "K_DOWN" in kb[PlayerAction.MOVE_DOWN]


class TestBuildKeyLookup:
    """Verify the reverse lookup builder."""

    def test_simple_binding(self) -> None:
        """A single action with one key produces a lookup entry."""
        bindings = {"mine": ["K_SPACE"]}
        lookup = build_key_lookup(bindings)
        assert (0, pygame.K_SPACE) in lookup
        assert "mine" in lookup[(0, pygame.K_SPACE)]

    def test_multiple_keys_same_action(self) -> None:
        """Every key bound to the same action resolves."""
        bindings = {"move_up": ["K_w", "K_UP"]}
        lookup = build_key_lookup(bindings)
        assert "move_up" in lookup[(0, pygame.K_w)]
        assert "move_up" in lookup[(0, pygame.K_UP)]

    def test_same_key_multiple_actions(self) -> None:
        """One key bound to different actions returns both actions."""
        bindings = {"move_up": ["K_w"], "nav_up": ["K_w"]}
        lookup = build_key_lookup(bindings)
        result = lookup[(0, pygame.K_w)]
        assert "move_up" in result
        assert "nav_up" in result

    def test_modifier_key(self) -> None:
        """SHIFT+key parses into a modified lookup entry."""
        bindings = {"slot_9": ["SHIFT+K_1"]}
        lookup = build_key_lookup(bindings)
        assert (pygame.KMOD_SHIFT, pygame.K_1) in lookup
        assert "slot_9" in lookup[(pygame.KMOD_SHIFT, pygame.K_1)]

    def test_ctrl_modifier(self) -> None:
        """CTRL+key parses into a modified lookup entry."""
        bindings = {"select_player_1": ["CTRL+K_1"]}
        lookup = build_key_lookup(bindings)
        assert (pygame.KMOD_CTRL, pygame.K_1) in lookup

    def test_invalid_key_name_skipped(self) -> None:
        """The parser logs an invalid key name and then skips it."""
        bindings = {"mine": ["K_FAKE_KEY"]}
        lookup = build_key_lookup(bindings)
        assert len(lookup) == 0

    def test_empty_bindings(self) -> None:
        """An empty key list produces no entries for that action."""
        bindings = {"turn_left": []}
        lookup = build_key_lookup(bindings)
        assert len(lookup) == 0


class TestResolveKey:
    """Verify key resolution with modifiers and fallback."""

    @pytest.fixture()
    def lookup(self) -> KeyLookup:
        """Build a lookup from default keyboard bindings."""
        return build_key_lookup(default_keyboard())

    def test_bare_key(self, lookup: KeyLookup) -> None:
        """A bare key press resolves to the matching actions."""
        result = resolve_key(lookup, pygame.K_w)
        assert PlayerAction.MOVE_UP in result
        assert PlayerAction.NAV_UP in result

    def test_shift_overrides_bare(self, lookup: KeyLookup) -> None:
        """SHIFT+1 matches slot_9, not slot_1."""
        result = resolve_key(lookup, pygame.K_1, pygame.KMOD_SHIFT)
        assert PlayerAction.SLOT_9 in result
        assert PlayerAction.SLOT_1 not in result

    def test_ctrl_overrides_bare(self, lookup: KeyLookup) -> None:
        """CTRL+1 matches select_player_1, not slot_1."""
        result = resolve_key(lookup, pygame.K_1, pygame.KMOD_CTRL)
        assert PlayerAction.SELECT_PLAYER_1 in result
        assert PlayerAction.SLOT_1 not in result

    def test_shift_on_unbound_key_falls_back(self, lookup: KeyLookup) -> None:
        """SHIFT+W has no specific binding, so it falls back to bare W."""
        result = resolve_key(lookup, pygame.K_w, pygame.KMOD_SHIFT)
        assert PlayerAction.MOVE_UP in result

    def test_unbound_key_returns_empty(self, lookup: KeyLookup) -> None:
        """A key with no binding returns an empty frozenset."""
        result = resolve_key(lookup, pygame.K_F12)
        assert len(result) == 0

    def test_escape_resolves_to_quit(self, lookup: KeyLookup) -> None:
        result = resolve_key(lookup, pygame.K_ESCAPE)
        assert PlayerAction.QUIT in result

    def test_backspace_resolves_to_back(self, lookup: KeyLookup) -> None:
        result = resolve_key(lookup, pygame.K_BACKSPACE)
        assert PlayerAction.BACK in result


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
