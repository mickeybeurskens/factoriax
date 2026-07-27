"""Launcher: the title menu and what each of its entries opens.

Owns the pygame window for the whole session and hands the same surface
to each sub-screen, so returning from play or the editor lands back on
the menu without a resize flicker. Sub-screens return control instead of
quitting; only :func:`run_app` tears pygame down.

The launcher is deliberately free-play only. Research scenarios are
built through :func:`factoriax.make` and driven by training code, not
through this UI.
"""

from __future__ import annotations

import pygame

from factoriax.playground.menu.main_menu import run_main_menu
from factoriax.playground.ui import theme as _theme
from factoriax.playground.ui.window import calculate_window_size

#: Menu window edge, in unscaled UI units.
_BASE_MENU_SIZE: int = 640


def run_app() -> None:
    """Open the launcher and loop until the user quits."""
    pygame.init()
    try:
        scaled = _BASE_MENU_SIZE * _theme.UI_SCALE
        width, height = calculate_window_size(scaled, scaled)
        screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("FactoriaX")

        while True:
            choice = run_main_menu(screen)
            if choice is None:
                return
            screen = _open(choice, screen)
    finally:
        pygame.quit()


def _open(choice: str, screen: pygame.Surface) -> pygame.Surface:
    """Run the chosen sub-screen and return the surface to keep using.

    Sub-screens may replace the display surface (the settings menu
    applies resolution and UI-scale changes), so the live surface is
    re-read from pygame on the way out rather than assumed unchanged.
    """
    if choice == "play":
        from factoriax.playground.play.main import main as play_main

        play_main(screen=screen)
    elif choice == "editor":
        from factoriax.playground.editor.main import main as editor_main

        editor_main(screen=screen)
    elif choice == "settings":
        from factoriax.playground.play.launch_screen import run_settings_menu

        run_settings_menu(screen)
    else:
        raise ValueError(f"unknown menu choice: {choice!r}")

    return pygame.display.get_surface() or screen
