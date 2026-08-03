"""Shared fixtures for the env tests.

``test_base.py`` and ``test_wrappers.py`` both build a plain 8x8 env and take
its default parameters. Module scope keeps the XLA compile to one per file.
"""

import pytest

from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.levels import Level, LevelBuilder
from factoriax.engine.state import EnvParams


@pytest.fixture(scope="module")
def level8() -> Level:
    """Return a plain 8x8 dirt level."""
    return LevelBuilder(8, 8).build("hooks")


@pytest.fixture(scope="module")
def params(level8: Level) -> EnvParams:
    """Return the default parameters of an env bound to ``level8``."""
    return FactoriaxEnv(level=level8).default_params
