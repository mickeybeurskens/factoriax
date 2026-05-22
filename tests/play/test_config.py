"""Tests for player configuration, bindings, and resolver."""

from __future__ import annotations

import time
from pathlib import Path

import orjson
import pygame
import pytest

from factoriax.config import (
    KeyLookup,
    PlayerAction,
    PlayerConfig,
    build_key_lookup,
    config_to_env_params,
    default_controller,
    default_keyboard,
    env_params_to_dict,
    load_config,
    resolve_key,
    save_config,
)
from factoriax.state import EnvParams

# Ensure pygame constants are available for key lookups.
pygame.init()


class TestDefaultBindings:
    """Verify default binding maps are complete and consistent."""

    def test_keyboard_covers_all_actions(self) -> None:
        """Every PlayerAction should appear in the default keyboard map."""
        kb = default_keyboard()
        for action in PlayerAction:
            assert action in kb, f"Missing keyboard binding for {action}"

    def test_controller_covers_gameplay_and_nav(self) -> None:
        """Controller defaults should cover movement and core navigation."""
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
        """Movement should be bound to both WASD and arrow keys."""
        kb = default_keyboard()
        assert "K_w" in kb[PlayerAction.MOVE_UP]
        assert "K_UP" in kb[PlayerAction.MOVE_UP]
        assert "K_s" in kb[PlayerAction.MOVE_DOWN]
        assert "K_DOWN" in kb[PlayerAction.MOVE_DOWN]


class TestBuildKeyLookup:
    """Verify the reverse lookup builder."""

    def test_simple_binding(self) -> None:
        """A single action with one key should produce a lookup entry."""
        bindings = {"mine": ["K_SPACE"]}
        lookup = build_key_lookup(bindings)
        assert (0, pygame.K_SPACE) in lookup
        assert "mine" in lookup[(0, pygame.K_SPACE)]

    def test_multiple_keys_same_action(self) -> None:
        """Multiple keys bound to the same action should all resolve."""
        bindings = {"move_up": ["K_w", "K_UP"]}
        lookup = build_key_lookup(bindings)
        assert "move_up" in lookup[(0, pygame.K_w)]
        assert "move_up" in lookup[(0, pygame.K_UP)]

    def test_same_key_multiple_actions(self) -> None:
        """One key bound to different actions should return both."""
        bindings = {"move_up": ["K_w"], "nav_up": ["K_w"]}
        lookup = build_key_lookup(bindings)
        result = lookup[(0, pygame.K_w)]
        assert "move_up" in result
        assert "nav_up" in result

    def test_modifier_key(self) -> None:
        """SHIFT+key should parse into a modified lookup entry."""
        bindings = {"slot_9": ["SHIFT+K_1"]}
        lookup = build_key_lookup(bindings)
        assert (pygame.KMOD_SHIFT, pygame.K_1) in lookup
        assert "slot_9" in lookup[(pygame.KMOD_SHIFT, pygame.K_1)]

    def test_ctrl_modifier(self) -> None:
        """CTRL+key should parse correctly."""
        bindings = {"select_player_1": ["CTRL+K_1"]}
        lookup = build_key_lookup(bindings)
        assert (pygame.KMOD_CTRL, pygame.K_1) in lookup

    def test_invalid_key_name_skipped(self) -> None:
        """Invalid key names should be logged and skipped."""
        bindings = {"mine": ["K_FAKE_KEY"]}
        lookup = build_key_lookup(bindings)
        assert len(lookup) == 0

    def test_empty_bindings(self) -> None:
        """Empty key list should produce no entries for that action."""
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
        """Bare key press should resolve to matching actions."""
        result = resolve_key(lookup, pygame.K_w)
        assert PlayerAction.MOVE_UP in result
        assert PlayerAction.NAV_UP in result

    def test_shift_overrides_bare(self, lookup: KeyLookup) -> None:
        """SHIFT+1 should match slot_9, not slot_1."""
        result = resolve_key(lookup, pygame.K_1, pygame.KMOD_SHIFT)
        assert PlayerAction.SLOT_9 in result
        assert PlayerAction.SLOT_1 not in result

    def test_ctrl_overrides_bare(self, lookup: KeyLookup) -> None:
        """CTRL+1 should match select_player_1, not slot_1."""
        result = resolve_key(lookup, pygame.K_1, pygame.KMOD_CTRL)
        assert PlayerAction.SELECT_PLAYER_1 in result
        assert PlayerAction.SLOT_1 not in result

    def test_shift_on_unbound_key_falls_back(self, lookup: KeyLookup) -> None:
        """SHIFT+W has no specific binding; should fall back to bare W."""
        result = resolve_key(lookup, pygame.K_w, pygame.KMOD_SHIFT)
        assert PlayerAction.MOVE_UP in result

    def test_unbound_key_returns_empty(self, lookup: KeyLookup) -> None:
        """A key with no binding should return an empty frozenset."""
        result = resolve_key(lookup, pygame.K_F12)
        assert len(result) == 0

    def test_escape_resolves_to_quit(self, lookup: KeyLookup) -> None:
        result = resolve_key(lookup, pygame.K_ESCAPE)
        assert PlayerAction.QUIT in result

    def test_backspace_resolves_to_back(self, lookup: KeyLookup) -> None:
        result = resolve_key(lookup, pygame.K_BACKSPACE)
        assert PlayerAction.BACK in result


class TestEnvParamsConversion:
    """Verify round-trip EnvParams <-> dict conversion."""

    def test_round_trip_defaults(self) -> None:
        """Default EnvParams should survive a dict round trip."""
        params = EnvParams()
        d = env_params_to_dict(params)
        restored = config_to_env_params(PlayerConfig(env_params=d))
        for field_name in d:
            assert getattr(restored, field_name) == getattr(params, field_name)

    def test_custom_values(self) -> None:
        """Custom param values should be preserved."""
        params = EnvParams(map_width=64, map_height=64, num_players=4)
        d = env_params_to_dict(params)
        assert d["map_width"] == 64
        assert d["num_players"] == 4

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields in the dict should fall back to defaults."""
        config = PlayerConfig(env_params={"map_width": 16})
        params = config_to_env_params(config)
        assert params.map_width == 16
        assert params.map_height == EnvParams().map_height

    def test_player_mining_yield_default(self) -> None:
        """EnvParams should expose a default player_mining_yield of 1."""
        assert EnvParams().player_mining_yield == 1

    def test_player_mining_yield_in_dict(self) -> None:
        """player_mining_yield should round-trip through env_params_to_dict."""
        d = env_params_to_dict(EnvParams())
        assert d["player_mining_yield"] == 1

    def test_player_mining_yield_custom_value(self) -> None:
        """A non-default player_mining_yield should survive round-trip."""
        params = EnvParams(player_mining_yield=5)
        d = env_params_to_dict(params)
        restored = config_to_env_params(PlayerConfig(env_params=d))
        assert d["player_mining_yield"] == 5
        assert restored.player_mining_yield == 5


class TestLoadSaveConfig:
    """Verify config persistence."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """Saving and loading should preserve all config fields."""
        path = tmp_path / "test_config.json"
        kb = default_keyboard()
        ctrl = default_controller()
        env_dict = env_params_to_dict(EnvParams(map_width=48))
        config = PlayerConfig(env_params=env_dict, keyboard=kb, controller=ctrl)

        save_config(config, path)
        loaded = load_config(path)

        assert loaded.env_params["map_width"] == 48
        assert loaded.keyboard[PlayerAction.MINE] == kb[PlayerAction.MINE]

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """Loading from a missing file should return default config."""
        path = tmp_path / "nonexistent.json"
        config = load_config(path)
        assert config.env_params["map_width"] == EnvParams().map_width
        assert PlayerAction.MINE in config.keyboard

    def test_load_merges_missing_actions(self, tmp_path: Path) -> None:
        """Saved config missing some actions should gain defaults."""
        path = tmp_path / "partial.json"
        import orjson

        partial = {
            "env_params": {"map_width": 20},
            "keyboard": {"mine": ["K_x"]},
            "controller": {},
        }
        path.write_bytes(orjson.dumps(partial))

        config = load_config(path)
        # Custom binding preserved.
        assert config.keyboard["mine"] == ["K_x"]
        # Missing action filled from defaults.
        assert PlayerAction.MOVE_UP in config.keyboard
        assert len(config.keyboard[PlayerAction.MOVE_UP]) > 0

    def test_load_corrupt_file_returns_defaults(self, tmp_path: Path) -> None:
        """Corrupt JSON should return defaults instead of crashing."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        config = load_config(path)
        assert config.env_params["map_width"] == EnvParams().map_width


class TestPlayerConfigSeed:
    """Verify the PlayerConfig.seed field and its persistence."""

    def test_default_seed_is_42(self) -> None:
        """PlayerConfig should default seed to 42."""
        assert PlayerConfig().seed == 42

    def test_seed_round_trip(self, tmp_path: Path) -> None:
        """save_config / load_config should preserve a custom seed."""
        path = tmp_path / "seed.json"
        config = PlayerConfig(
            env_params=env_params_to_dict(EnvParams()),
            keyboard=default_keyboard(),
            controller=default_controller(),
            seed=12345,
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.seed == 12345

    def test_seed_large_int_round_trip(self, tmp_path: Path) -> None:
        """Time.time_ns()-magnitude ints (~10^18) must survive round-trip."""
        path = tmp_path / "big_seed.json"
        large_seed = time.time_ns()
        assert large_seed > 10**18
        config = PlayerConfig(
            env_params=env_params_to_dict(EnvParams()),
            keyboard=default_keyboard(),
            controller=default_controller(),
            seed=large_seed,
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.seed == large_seed
        # Confirm the on-disk value is a plain int, not a float coercion.
        raw = orjson.loads(path.read_bytes())
        assert isinstance(raw["seed"], int)
        assert raw["seed"] == large_seed

    def test_missing_seed_falls_back_to_default(self, tmp_path: Path) -> None:
        """A config file without `seed` should load to seed=42."""
        path = tmp_path / "no_seed.json"
        path.write_bytes(orjson.dumps({"env_params": {}, "keyboard": {}}))
        loaded = load_config(path)
        assert loaded.seed == 42

    def test_load_missing_file_returns_default_seed(self, tmp_path: Path) -> None:
        """Missing config file should yield seed=42."""
        path = tmp_path / "nope.json"
        loaded = load_config(path)
        assert loaded.seed == 42
