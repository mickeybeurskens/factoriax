"""Tests for the :class:`EnvState` schema in :mod:`factoriax.engine.state`.

The state is a flax struct, so its field set is a contract. Every consumer
reads it by name: the observation encoder, the renderer, the recorder, and
the playground. A field that changes shape or dtype breaks them silently.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from factoriax.engine.state import EnvParams


class TestEnvStateSchema:
    """Tests for the EnvState schema: the fields the engine guarantees."""

    def test_generate_state_initializes_achievements_unlocked(self) -> None:
        """Procedural state has all-False achievements_unlocked of correct shape."""
        from factoriax.engine.constants import MAX_ACHIEVEMENTS
        from factoriax.engine.levels import generate_state

        state = generate_state(random.PRNGKey(0), EnvParams())

        assert state.achievements_unlocked.shape == (MAX_ACHIEVEMENTS,)
        assert state.achievements_unlocked.dtype == jnp.bool_
        assert not bool(state.achievements_unlocked.any())
