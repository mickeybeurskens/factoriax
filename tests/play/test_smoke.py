"""Smoke tests for play and editor modules.

These tests verify that the play UI and editor can import, create objects,
and render frames without crashing. They do NOT test game logic correctness
or visual output quality — just that the code paths don't hit missing
attributes, import errors, or type mismatches after state changes.

Pygame fonts are initialized by the session-scoped ``pygame_font_session``
fixture in ``tests/play/conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.config import build_key_lookup, default_keyboard
from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams


@pytest.fixture(scope="module")
def env_and_state():
    """Create a small environment and state for smoke tests."""
    env = FactoriaXEnv()
    params = EnvParams(map_width=16, map_height=16, num_players=1)
    _, state = env.reset_env(jax.random.key(42), params)
    return env, params, state


class TestRendererSmoke:
    """Renderer should produce an image from any valid state."""

    def test_render_pixels(self, env_and_state) -> None:
        """render_pixels returns an RGB array."""
        from factoriax.renderer import render_pixels

        _, _, state = env_and_state
        img = render_pixels(state, block_pixel_size=8)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_render_after_step(self, env_and_state) -> None:
        """Rendering works after stepping the environment."""
        from factoriax.renderer import render_pixels

        env, params, state = env_and_state
        _, state2, _, _, _ = env.step_env(
            jax.random.key(1),
            state,
            jnp.int32(Action.NOOP),
            params,
        )
        img = render_pixels(state2, block_pixel_size=8)
        assert img.shape == (128, 128, 3)


class TestPlayUISmoke:
    """Play UI rendering should not crash on a fresh state."""

    def test_inventory_menu(self, env_and_state) -> None:
        """render_inventory_menu returns an RGBA overlay."""
        from factoriax.play.ui import render_inventory_menu

        _, _, state = env_and_state
        overlay, regions = render_inventory_menu(state, 480, 480)
        assert overlay.shape == (480, 480, 4)
        assert overlay.dtype == np.uint8

    def test_achievement_menu(self) -> None:
        """render_achievement_menu returns an RGBA overlay."""
        from factoriax.constants import MAX_ACHIEVEMENTS
        from factoriax.play.ui import render_achievement_menu

        achievements = np.zeros(MAX_ACHIEVEMENTS, dtype=np.bool_)
        overlay = render_achievement_menu(achievements, 480, 480)
        assert overlay.shape == (480, 480, 4)

    def test_welcome_screen(self) -> None:
        """render_welcome_screen returns an overlay and regions."""
        from factoriax.play.ui import render_welcome_screen

        overlay, regions = render_welcome_screen(480, 480, False)
        assert overlay.shape == (480, 480, 4)

    def test_game_ui_render_frame(self, env_and_state) -> None:
        """GameUI.render_frame produces an RGB frame."""
        from factoriax.play.game_ui import GameUI

        _, params, state = env_and_state
        kb = build_key_lookup(default_keyboard())
        ui = GameUI(params, kb)
        frame, regions = ui.render_frame(state, 480, 480, 8, 0, 0)
        assert frame.shape == (480, 480, 3)
        assert frame.dtype == np.uint8


class TestConfigSmoke:
    """Config loading and EnvParams conversion should work."""

    def test_env_params_to_dict(self) -> None:
        """env_params_to_dict covers all current EnvParams fields."""
        from factoriax.config import env_params_to_dict

        d = env_params_to_dict(EnvParams())
        assert isinstance(d, dict)
        assert "map_width" in d
        assert "max_timesteps" in d

    def test_env_params_round_trip(self) -> None:
        """env_params_to_dict values match EnvParams fields."""
        from factoriax.config import env_params_to_dict

        params = EnvParams(map_width=64, map_height=64)
        d = env_params_to_dict(params)
        assert d["map_width"] == 64
        assert d["map_height"] == 64

    def test_load_config(self) -> None:
        """load_config returns a PlayerConfig without crashing."""
        from factoriax.config import load_config

        config = load_config()
        assert config is not None

    def test_settings_fields_exist_on_env_params(self) -> None:
        """Every field in the settings menu must exist on EnvParams."""
        from factoriax.menu.settings_menu import _build_sections

        params = EnvParams()
        sections = _build_sections(params)
        for section in sections:
            for field in section.fields:
                assert hasattr(params, field.name), (
                    f"Settings field {field.name!r} not found on EnvParams"
                )

    def test_env_param_fields_match_env_params(self) -> None:
        """_ENV_PARAM_FIELDS in config.py must all exist on EnvParams."""
        from factoriax.config import _ENV_PARAM_FIELDS

        params = EnvParams()
        for name in _ENV_PARAM_FIELDS:
            assert hasattr(params, name), (
                f"_ENV_PARAM_FIELDS entry {name!r} not on EnvParams"
            )


class TestEditorSmoke:
    """Editor modules should import and create basic objects."""

    def test_editor_imports(self) -> None:
        """Editor dialogs and state modules import without error."""
        from factoriax.editor import (
            dialogs,  # noqa: F401
            state,  # noqa: F401
            toolbar,  # noqa: F401
        )


@pytest.mark.slow
class TestEnvStepSmoke:
    """Environment should reset and step at various map sizes.

    Every method triggers a fresh JIT compile for the env at a given
    map size. Slow but load-bearing — gated behind ``-m slow``.
    """

    @pytest.mark.parametrize("size", [16, 32, 64])
    def test_reset_and_step(self, size: int) -> None:
        """Reset + step completes without error at each map size."""
        env = FactoriaXEnv()
        params = EnvParams(
            map_width=size,
            map_height=size,
            num_players=1,
        )
        _, state = env.reset_env(jax.random.key(0), params)
        obs, state2, reward, done, info = env.step_env(
            jax.random.key(1),
            state,
            jnp.int32(Action.NOOP),
            params,
        )
        assert state2.timestep == 1
        assert obs.shape[0] > 0

    def test_vmapped_step(self) -> None:
        """Vmapped step works across a batch."""
        env = FactoriaXEnv()
        params = EnvParams(map_width=16, map_height=16, num_players=1)
        n = 8
        keys = jax.random.split(jax.random.key(42), n)
        _, states = jax.vmap(env.reset_env, in_axes=(0, None))(
            keys,
            params,
        )
        vmap_step = jax.jit(
            jax.vmap(env.step_env, in_axes=(0, 0, 0, None)),
        )
        step_keys = jax.random.split(jax.random.key(99), n)
        actions = jnp.zeros(n, dtype=jnp.int32)
        _, states2, _, _, _ = vmap_step(
            step_keys,
            states,
            actions,
            params,
        )
        assert states2.timestep.shape == (n,)
