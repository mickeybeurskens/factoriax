"""Headless integration tests for the play settings menu.

Drives :func:`factoriax.play.launch_screen.run_settings_menu` via a
prebuilt event queue so the menu's text rendering, signature, and
return value can be asserted without an interactive window.
"""

from __future__ import annotations

import pygame

from factoriax.config import (
    PlayerConfig,
    default_controller,
    default_keyboard,
    env_params_to_dict,
)
from factoriax.play.launch_screen import (
    _build_sections,
    _Button,
    run_settings_menu,
)
from factoriax.state import EnvParams


def _make_config(seed: int = 42, player_mining_yield: int = 1) -> PlayerConfig:
    """Build a PlayerConfig with the given seed and yield."""
    env = env_params_to_dict(EnvParams(player_mining_yield=player_mining_yield))
    return PlayerConfig(
        env_params=env,
        seed=seed,
        keyboard=default_keyboard(),
        controller=default_controller(),
    )


class TestSectionLayout:
    """Static checks on the section / field layout."""

    def test_seed_field_present(self) -> None:
        """The Seed field should be in the section layout."""
        sections = _build_sections(_make_config())
        names = [fs.name for sec in sections for fs in sec.fields]
        assert "seed" in names

    def test_player_mining_yield_field_present(self) -> None:
        """player_mining_yield should be in the section layout."""
        sections = _build_sections(_make_config())
        names = [fs.name for sec in sections for fs in sec.fields]
        assert "player_mining_yield" in names

    def test_machine_mining_rate_label(self) -> None:
        """miner_mining_rate should be labelled 'Machine Mining Rate'."""
        sections = _build_sections(_make_config())
        labels = {fs.name: fs.label for sec in sections for fs in sec.fields}
        assert labels["miner_mining_rate"] == "Machine Mining Rate"

    def test_player_mining_yield_label(self) -> None:
        """player_mining_yield should be labelled 'Player Mining Yield'."""
        sections = _build_sections(_make_config())
        labels = {fs.name: fs.label for sec in sections for fs in sec.fields}
        assert labels["player_mining_yield"] == "Player Mining Yield"

    def test_no_bare_mining_rate_label(self) -> None:
        """The bare label 'Mining Rate' (no qualifier) must be gone."""
        sections = _build_sections(_make_config())
        labels = [fs.label for sec in sections for fs in sec.fields]
        assert "Mining Rate" not in labels


class TestRandomizeButton:
    """Verify the Randomize button on the Seed field."""

    def test_randomize_button_assigns_positive_time_ns(self) -> None:
        """Clicking Randomize should write a positive int to the seed field."""
        config = _make_config(seed=42)
        sections = _build_sections(config)
        seed_field = next(
            fs for sec in sections for fs in sec.fields if fs.name == "seed"
        )
        seed_field.editing = True
        seed_field.edit_buffer = "42"

        button = _Button(label="Randomize", target_field=seed_field)
        button.on_click()
        assert seed_field.edit_buffer.isdigit()
        assert int(seed_field.edit_buffer) > 0
        assert int(seed_field.edit_buffer) != 42


class TestRunSettingsMenu:
    """End-to-end pygame drive of run_settings_menu."""

    def test_menu_returns_player_config(self) -> None:
        """Closing the menu via Escape returns a PlayerConfig."""
        initial = _make_config(seed=12345, player_mining_yield=2)
        # Drive: post ESCAPE so the menu exits immediately.
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        screen = pygame.display.set_mode((640, 640))
        result = run_settings_menu(screen, initial_config=initial)

        assert isinstance(result, PlayerConfig)
        assert result.seed == 12345
        assert result.env_params["player_mining_yield"] == 2

    def test_menu_save_called_on_close(self, monkeypatch, tmp_path) -> None:
        """save_config should fire on Escape exit (commit-on-close)."""
        initial = _make_config()
        calls: list[PlayerConfig] = []

        def fake_save(cfg: PlayerConfig, path=None) -> None:
            calls.append(cfg)

        monkeypatch.setattr("factoriax.play.launch_screen.save_config", fake_save)

        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        screen = pygame.display.set_mode((640, 640))
        run_settings_menu(screen, initial_config=initial)

        assert calls, "save_config was not invoked on Escape close"
