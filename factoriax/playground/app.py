"""Launcher: the title menu and what each of its entries opens.

The launcher holds the pygame window for the whole session. It gives the
same surface to each sub-screen, so a return from play or from the editor
comes back to the menu with no flicker.

A sub-screen returns control, and does not quit. :func:`run_app` closes
pygame, and no other function does.

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
        pygame.display.set_caption("Factoriax")

        while True:
            choice = run_main_menu(screen)
            if choice is None:
                return
            screen = _open(choice, screen)
    finally:
        pygame.quit()


def _open(choice: str, screen: pygame.Surface) -> pygame.Surface:
    """Run the chosen sub-screen and return the surface to keep using.

    A sub-screen can replace the display surface, because the settings menu
    applies a new resolution and a new UI scale. The function therefore reads
    the surface back from pygame on the way out.
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
