"""Scripted baseline for the L.1 navigate skill.

The level places a single coal tile at a seed-varying position; the
player must walk onto it. The policy reads the terrain map, finds
the nearest mineable tile, and steps toward it greedily (manhattan).
``MINE`` is blocked by the per-level mask, so the policy never tries
to mine — it just walks until the achievement (player on a mineable
tile) latches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import Action, BlockType
from factoriax.engine.state import EnvParams, EnvState

_ORE_BLOCK_VALUES: tuple[int, ...] = (
    int(BlockType.COAL),
    int(BlockType.IRON),
    int(BlockType.COPPER),
    int(BlockType.TIN),
    int(BlockType.SILICON),
)


def _action(a: int) -> jax.Array:
    """Wrap an int as a JAX int32 scalar."""
    out: jax.Array = jnp.asarray(a, dtype=jnp.int32)
    return out


def navigate_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Walk greedy-manhattan toward the nearest mineable tile.

    Args:
        state: Current environment state.
        params: Environment parameters (unused).

    Returns:
        JAX int32 scalar action.
    """
    del params
    map_arr = np.asarray(state.map)
    pos = np.asarray(state.player_positions[0])
    px, py = int(pos[0]), int(pos[1])

    if int(map_arr[py, px]) in _ORE_BLOCK_VALUES:
        return _action(int(Action.NOOP))

    ys, xs = np.where(np.isin(map_arr, _ORE_BLOCK_VALUES))
    if xs.size == 0:
        return _action(int(Action.NOOP))
    distances = np.abs(xs - px) + np.abs(ys - py)
    closest = int(np.argmin(distances))
    target_x, target_y = int(xs[closest]), int(ys[closest])

    if px < target_x:
        return _action(int(Action.RIGHT))
    if px > target_x:
        return _action(int(Action.LEFT))
    if py < target_y:
        return _action(int(Action.DOWN))
    if py > target_y:
        return _action(int(Action.UP))
    return _action(int(Action.NOOP))
