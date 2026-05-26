"""Tests for the launch screen."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from factoriax.config import (  # noqa: E402
    PlayerConfig,
    default_controller,
    default_keyboard,
    env_params_to_dict,
)
from factoriax.engine.state import EnvParams  # noqa: E402
from factoriax.play.launch_screen import (  # noqa: E402
    _PAGE_OPTIONS,
    _SETTING_FIELDS,
    _get_value,
    _reset_to_defaults,
    _set_value,
    run_settings_menu,
)


def _make_config(seed: int = 42, player_mining_yield: int = 1) -> PlayerConfig:
    env = env_params_to_dict(EnvParams(player_mining_yield=player_mining_yield))
    return PlayerConfig(
        env_params=env,
        seed=seed,
        keyboard=default_keyboard(),
        controller=default_controller(),
    )


class TestSettingFields:
    def test_seed_field_present(self) -> None:
        keys = [f.key for f in _SETTING_FIELDS]
        assert "seed" in keys

    def test_required_env_params_present(self) -> None:
        required = {
            "map_width",
            "map_height",
            "num_players",
            "max_timesteps",
            "water_probability",
            "iron_probability",
            "copper_probability",
            "coal_probability",
            "tin_probability",
            "silicon_probability",
            "base_resources",
            "max_machines",
            "miner_mining_rate",
            "player_mining_yield",
        }
        keys = {f.key for f in _SETTING_FIELDS}
        assert required.issubset(keys)

    def test_page_options_are_play_settings_reset(self) -> None:
        assert [o.action for o in _PAGE_OPTIONS] == ["play", "settings", "reset"]


class TestValueOps:
    def test_get_value_reads_seed_from_top_level(self) -> None:
        config = _make_config(seed=99)
        seed_field = next(f for f in _SETTING_FIELDS if f.key == "seed")
        assert int(_get_value(config, seed_field)) == 99

    def test_set_value_writes_seed_back(self) -> None:
        config = _make_config(seed=1)
        seed_field = next(f for f in _SETTING_FIELDS if f.key == "seed")
        _set_value(config, seed_field, 7.0)
        assert config.seed == 7

    def test_set_value_casts_int_for_int_field(self) -> None:
        config = _make_config()
        field = next(f for f in _SETTING_FIELDS if f.key == "map_width")
        _set_value(config, field, 17.0)
        assert config.env_params["map_width"] == 17
        assert isinstance(config.env_params["map_width"], int)

    def test_field_clamp_respects_min_max_for_int(self) -> None:
        field = next(f for f in _SETTING_FIELDS if f.key == "num_players")
        assert field.clamp(0) == field.min_value
        assert field.clamp(99) == field.max_value

    def test_field_clamp_rounds_float_to_two_decimals(self) -> None:
        field = next(f for f in _SETTING_FIELDS if f.key == "iron_probability")
        assert field.clamp(0.123456) == 0.12


class TestResetToDefaults:
    def test_replaces_env_params_with_defaults(self) -> None:
        config = _make_config(player_mining_yield=9)
        _reset_to_defaults(config)
        defaults = env_params_to_dict(EnvParams())
        for key, default in defaults.items():
            assert config.env_params[key] == default

    def test_picks_new_seed(self) -> None:
        config = _make_config(seed=42)
        _reset_to_defaults(config)
        assert config.seed != 42


class TestRunSettingsMenu:
    def test_menu_returns_none_on_backspace_cancel(self) -> None:
        initial = _make_config(seed=12345, player_mining_yield=2)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))

        screen = pygame.display.set_mode((640, 640))
        result = run_settings_menu(screen, initial_config=initial)

        assert result is None

    def test_menu_save_called_on_close(self, monkeypatch) -> None:
        initial = _make_config()
        calls: list[PlayerConfig] = []

        def fake_save(cfg: PlayerConfig, path=None) -> None:
            calls.append(cfg)

        monkeypatch.setattr("factoriax.play.launch_screen.save_config", fake_save)

        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))

        screen = pygame.display.set_mode((640, 640))
        run_settings_menu(screen, initial_config=initial)

        assert calls, "save_config was not invoked on close"
