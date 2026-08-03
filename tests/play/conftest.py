"""Give every play test a display surface and a font subsystem.

The play UI renders. The root ``pygame_display`` fixture is not autouse, so
this file turns it on for this directory alone.
"""

import pytest


@pytest.fixture(autouse=True)
def _display(pygame_display: None) -> None:
    """Request the session display for every test in this directory."""
