"""Entry point for ``python -m factoriax``.

Shows the main menu, then launches Play (with settings), Editor, or
Settings based on the user's choice. Player configuration is loaded
once at startup and persisted on play.
"""

from __future__ import annotations

from typing import Any, cast

import pygame

from factoriax.config import (
    PlayerConfig,
    build_controller_lookup,
    build_key_lookup,
    config_to_env_params,
    env_params_to_dict,
    load_config,
    save_config,
)
from factoriax.ui import theme as _theme
from factoriax.ui.window import auto_ui_scale, calculate_window_size

_BASE_SIZE = 1024


def _run() -> None:
    """Main menu loop: show menu, dispatch to play, editor, or settings."""
    pygame.init()

    config = load_config()
    ui_scale = config.ui_scale if config.ui_scale > 0 else auto_ui_scale()
    _theme.apply_scale(ui_scale)

    canvas_size = _BASE_SIZE * ui_scale
    if config.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        w, h = calculate_window_size(canvas_size, canvas_size)
        screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("FactoriaX")

    from factoriax.menu.main_menu import run_main_menu

    kb_lookup = build_key_lookup(config.keyboard)
    ctrl_lookup = build_controller_lookup(config.controller)

    while True:
        choice = run_main_menu(screen, kb_lookup, ctrl_lookup)

        if choice is None:
            break

        if choice == "play":
            _handle_play(screen, config)
        elif choice == "editor":
            _handle_editor(screen)
        elif choice == "settings":
            _handle_settings(screen, config)
            # Rebuild window in case scale or fullscreen changed.
            screen = _apply_display_config(config)

    pygame.quit()


def _apply_display_config(config: PlayerConfig) -> pygame.Surface:
    """Apply display settings from config and return the new screen.

    Reapplies the theme scale and recreates the pygame display at the
    correct size and mode.

    Args:
        config: Player configuration with display settings.

    Returns:
        The new pygame display surface.
    """
    ui_scale = config.ui_scale if config.ui_scale > 0 else auto_ui_scale()
    _theme.apply_scale(ui_scale)
    canvas_size = _BASE_SIZE * ui_scale
    if config.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        w, h = calculate_window_size(canvas_size, canvas_size)
        screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("FactoriaX")
    return screen


def _handle_play(screen: pygame.Surface, config: PlayerConfig) -> None:
    """Show play settings, then launch the game with chosen parameters."""
    from factoriax.play.launch_screen import run_settings_menu

    initial_params = config_to_env_params(config)
    params = run_settings_menu(screen, initial_params=initial_params)
    if params is None:
        return

    config.env_params = env_params_to_dict(params)
    save_config(config)

    from jax import random

    from factoriax.achievements import core_game_conditions
    from factoriax.envs.factoriax_env import FactoriaXEnv
    from factoriax.play.main import _play_loop, _run_with_loading_screen

    env = FactoriaXEnv(achievement_fn=core_game_conditions)

    rng = random.PRNGKey(42)
    rng, reset_key = random.split(rng)
    _reset_key = reset_key

    # ``_run_with_loading_screen`` is generic over the worker's return
    # type but typed as ``object``; cast to the concrete ``(obs, state)``
    # tuple ``env.reset_env`` actually returns.
    reset_result = cast(
        "tuple[Any, Any]",
        _run_with_loading_screen(
            screen,
            "Generating world",
            lambda: env.reset_env(_reset_key, params),
        ),
    )
    _, state = reset_result

    kb_lookup = build_key_lookup(config.keyboard)
    ctrl_lookup = build_controller_lookup(config.controller)
    pygame.display.set_caption("FactoriaX")
    _play_loop(
        env,
        state,
        params,
        screen,
        rng,
        kb_lookup=kb_lookup,
        ctrl_lookup=ctrl_lookup,
    )


def _handle_editor(screen: pygame.Surface) -> None:
    """Launch the level editor."""
    from factoriax.editor.main import main as editor_main

    editor_main(screen=screen)
    pygame.display.set_caption("FactoriaX")


def _handle_settings(screen: pygame.Surface, config: PlayerConfig) -> None:
    """Open the controls/rebinding screen and persist changes."""
    from factoriax.menu.controls_menu import run_controls_menu

    new_fullscreen, new_scale = run_controls_menu(screen, config)
    config.fullscreen = new_fullscreen
    config.ui_scale = new_scale
    save_config(config)


if __name__ == "__main__":
    _run()
