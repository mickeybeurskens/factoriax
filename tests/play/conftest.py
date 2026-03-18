"""Shared pytest configuration for the play test package.

Manages the pygame font subsystem lifecycle at session scope so that
multiple test modules can share it without triggering init/quit cycles
that cause segfaults.
"""

import pygame
import pytest


@pytest.fixture(scope="session", autouse=True)
def pygame_font_session() -> None:
    """Initialise the pygame font subsystem once per test session."""
    pygame.font.init()
    yield
    pygame.font.quit()
