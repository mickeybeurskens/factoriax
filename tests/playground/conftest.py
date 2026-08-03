"""Give every playground test a display surface and a font subsystem.

The playground renders. The root ``pygame_display`` fixture is not autouse, so
this file turns it on for this directory tree alone. Most of the suite never
touches pygame, and an unconditional display init taxes every one of those
tests.
"""

import pytest


@pytest.fixture(autouse=True)
def _display(pygame_display: None) -> None:
    """Request the session display for every test under this directory."""
