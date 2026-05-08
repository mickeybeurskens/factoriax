"""Scripted baseline for the L.2 mine skill.

If the player is standing on an ore tile, ``MINE``. Otherwise step
greedily toward the nearest ore tile (manhattan, no obstacle planning
required — ore tiles are walkable). NOOP only if the map has no ore
at all (degenerate case the level builder shouldn't produce).

Reads the terrain map and inventory directly from :class:`EnvState`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.constants import Action, BlockType
from factoriax.state import EnvParams, EnvState

_ORE_BLOCK_VALUES: tuple[int, ...] = (
    int(BlockType.COAL),
    int(BlockType.IRON),
    int(BlockType.COPPER),
    int(BlockType.TIN),
    int(BlockType.SILICON),
)


def _action(a: int) -> jax.Array:
    """Wrap an int as a JAX int32 scalar (mypy-friendly)."""
    out: jax.Array = jnp.asarray(a, dtype=jnp.int32)
    return out


def mine_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Walk to the nearest ore tile, then ``MINE``.

    Conversion to numpy is deliberate: the policy reads
    ``state.map`` and ``state.player_positions`` once per tick — JIT
    is unnecessary and the numpy path keeps the code readable as a
    tutorial example. The output is still a JAX scalar so the runner's
    ``jit_step`` accepts it.

    Args:
        state: Current environment state.
        params: Environment parameters (unused; kept for the
            scripted-policy signature).

    Returns:
        JAX int32 scalar action.
    """
    del params
    map_arr = np.asarray(state.map)
    pos = np.asarray(state.player_positions[0])
    px, py = int(pos[0]), int(pos[1])

    if int(map_arr[py, px]) in _ORE_BLOCK_VALUES:
        return _action(int(Action.MINE))

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
