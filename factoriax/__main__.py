"""Entry point for ``python -m factoriax``.

Shows the main menu, then launches Play (with settings) or Editor
based on the user's choice. The menu loop repeats until the user
quits.
"""

from __future__ import annotations

import pygame

from factoriax.ui.window import calculate_window_size

_UI_SIZE = 1024


def _run() -> None:
    """Main menu loop: show menu, dispatch to play or editor, repeat."""
    pygame.init()
    w, h = calculate_window_size(_UI_SIZE, _UI_SIZE)
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("FactoriaX")

    from factoriax.menu.main_menu import run_main_menu

    while True:
        choice = run_main_menu(screen)

        if choice is None:
            break

        if choice == "play":
            _handle_play(screen)

        elif choice == "editor":
            _handle_editor(screen)

    pygame.quit()


def _handle_play(screen: pygame.Surface) -> None:
    """Show settings, then launch the game with chosen parameters."""
    from factoriax.menu.settings_menu import run_settings_menu

    params = run_settings_menu(screen)
    if params is None:
        return

    import jax
    import jax.numpy as jnp
    from jax import random

    from factoriax.constants import Action
    from factoriax.envs.factoriax_env import make_factoriax_env
    from factoriax.play.main import _play_loop, _run_with_loading_screen
    from factoriax.state import EnvState

    env, _ = make_factoriax_env()

    rng = random.PRNGKey(42)
    rng, reset_key = random.split(rng)
    _reset_key = reset_key

    reset_result = _run_with_loading_screen(
        screen,
        "Generating world",
        lambda: env.reset_env(_reset_key, params),
    )
    state: EnvState = reset_result[1]  # type: ignore[index]

    step_fn = jax.jit(env.step_env)
    rng, warmup_key = random.split(rng)
    _warmup_key = warmup_key

    def _warmup() -> None:
        step_fn(
            _warmup_key, state, jnp.int32(Action.NOOP), params,
        )[0].block_until_ready()

    _run_with_loading_screen(
        screen, "Compiling JAX (first run only)", _warmup,
    )

    pygame.display.set_caption("FactoriaX")
    _play_loop(env, state, params, None, screen, rng)


def _handle_editor(screen: pygame.Surface) -> None:
    """Launch the level editor."""
    from factoriax.editor.main import main as editor_main

    editor_main(screen=screen)
    pygame.display.set_caption("FactoriaX")


if __name__ == "__main__":
    _run()
