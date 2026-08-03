"""Tests for reading and writing a player configuration.

``save_config`` and ``load_config`` round trip a :class:`PlayerConfig` through
an orjson file. A seed keeps a fresh install usable, and a mutation of the
binding table must survive the round trip.
"""

from __future__ import annotations

import time
from pathlib import Path

import orjson
import pygame

from factoriax.engine.state import EnvParams
from factoriax.playground.config import (
    PlayerAction,
    PlayerConfig,
    build_controller_lookup,
    build_key_lookup,
    default_controller,
    default_keyboard,
    env_params_to_dict,
    load_config,
    resolve_controller_button,
    resolve_key,
    save_config,
)

# Ensure pygame constants are available for key lookups.
pygame.init()


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
