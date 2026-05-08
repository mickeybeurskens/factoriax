"""Scripted baseline for the L.1 navigate skill.

Walks the player toward the bottom-right corner using a greedy
manhattan policy: step right while not yet at the right edge, then
step down. Reads ``state.player_positions`` directly — scripted
baselines are state-readers, not obs-readers.

This is the simplest possible solver for the simplest possible level
in the curriculum. It exists for three reasons: (1) prove the level
is solvable, (2) provide a deterministic score floor for RL to beat,
(3) demonstrate the per-skill scripted baseline shape that L.2-L.8
will follow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import Action
from factoriax.state import EnvParams, EnvState


def navigate_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Greedy manhattan walk to the bottom-right corner.

    Reads the player's ``(x, y)`` position from
    ``state.player_positions[0]`` and returns ``RIGHT`` until at the
    right edge, then ``DOWN`` until at the bottom. ``NOOP`` after that
    (the achievement should already have triggered, so the runner will
    stop tallying anyway).

    Args:
        state: Current environment state.
        params: Environment parameters (used for map dimensions).

    Returns:
        JAX int32 scalar action.
    """
    pos = state.player_positions[0]
    px, py = pos[0], pos[1]
    goal_x = params.map_width - 1
    goal_y = params.map_height - 1
    action: jax.Array = jnp.where(
        px < goal_x,
        jnp.int32(int(Action.RIGHT)),
        jnp.where(
            py < goal_y,
            jnp.int32(int(Action.DOWN)),
            jnp.int32(int(Action.NOOP)),
        ),
    )
    return action
