"""Tests for player configuration, bindings, and resolver."""

from __future__ import annotations

import time
from pathlib import Path

import orjson
import pygame
import pytest

from factoriax.engine.state import EnvParams
from factoriax.playground.config import (
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


class TestEnvParamsConversion:
    """Verify round-trip EnvParams <-> dict conversion."""

    def test_round_trip_defaults(self) -> None:
        """Default EnvParams survives a dict round trip."""
        params = EnvParams()
        d = env_params_to_dict(params)
        restored = config_to_env_params(PlayerConfig(env_params=d))
        for field_name in d:
            assert getattr(restored, field_name) == getattr(params, field_name)

    def test_custom_values(self) -> None:
        """A round trip keeps custom param values."""
        params = EnvParams(player_mining_yield=5)
        d = env_params_to_dict(params)
        assert d["player_mining_yield"] == 5

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields in the dict fall back to defaults."""
        config = PlayerConfig(env_params={"max_timesteps": 500})
        params = config_to_env_params(config)
        assert params.max_timesteps == 500
        assert params.player_mining_yield == EnvParams().player_mining_yield

    def test_player_mining_yield_default(self) -> None:
        """EnvParams exposes a default player_mining_yield of 1."""
        assert EnvParams().player_mining_yield == 1

    def test_player_mining_yield_in_dict(self) -> None:
        """player_mining_yield round-trips through env_params_to_dict."""
        d = env_params_to_dict(EnvParams())
        assert d["player_mining_yield"] == 1

    def test_player_mining_yield_custom_value(self) -> None:
        """A non-default player_mining_yield survives the round trip."""
        params = EnvParams(player_mining_yield=5)
        d = env_params_to_dict(params)
        restored = config_to_env_params(PlayerConfig(env_params=d))
        assert d["player_mining_yield"] == 5
        assert restored.player_mining_yield == 5


class TestLoadSaveConfig:
    """Verify config persistence."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """A save and a load keep every config field."""
        path = tmp_path / "test_config.json"
        kb = default_keyboard()
        ctrl = default_controller()
        env_dict = env_params_to_dict(EnvParams(player_mining_yield=3))
        config = PlayerConfig(env_params=env_dict, keyboard=kb, controller=ctrl)

        save_config(config, path)
        loaded = load_config(path)

        assert loaded.env_params["player_mining_yield"] == 3
        assert loaded.keyboard[PlayerAction.MINE] == kb[PlayerAction.MINE]

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """A load from a missing file returns the default config."""
        path = tmp_path / "nonexistent.json"
        config = load_config(path)
        assert config.env_params["max_timesteps"] == EnvParams().max_timesteps
        assert PlayerAction.MINE in config.keyboard

    def test_load_merges_missing_actions(self, tmp_path: Path) -> None:
        """A saved config that lacks some actions gains the defaults."""
        path = tmp_path / "partial.json"
        import orjson

        partial = {
            "env_params": {"max_timesteps": 500},
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
        """Corrupt JSON returns the defaults instead of a crash."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        config = load_config(path)
        assert config.env_params["max_timesteps"] == EnvParams().max_timesteps


class TestPlayerConfigSeed:
    """Verify the PlayerConfig.seed field and its persistence."""

    def test_default_seed_is_42(self) -> None:
        """PlayerConfig defaults the seed to 42."""
        assert PlayerConfig().seed == 42

    def test_seed_round_trip(self, tmp_path: Path) -> None:
        """save_config and load_config keep a custom seed."""
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
        """A config file without `seed` loads to seed=42."""
        path = tmp_path / "no_seed.json"
        path.write_bytes(orjson.dumps({"env_params": {}, "keyboard": {}}))
        loaded = load_config(path)
        assert loaded.seed == 42

    def test_load_missing_file_returns_default_seed(self, tmp_path: Path) -> None:
        """A missing config file gives seed=42."""
        path = tmp_path / "nope.json"
        loaded = load_config(path)
        assert loaded.seed == 42
