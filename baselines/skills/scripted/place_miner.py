"""Scripted baseline for the L.4 place_miner skill.

The level holds a 2x2 ore patch in one of four corners; the player
spawns at the centre with 5 miners in inventory. The policy walks
toward the patch, stops one tile short, faces the patch, and emits
``PLACE_MINER`` — placement happens on the tile in front of the
player. The achievement (``count_miners_on_ore >= 1``) latches as
soon as the placed miner lands on a mineable tile.

The patch is reachable from spawn in <=4 manhattan steps (centre is
at (2, 2); nearest patch tile is at (2, 0), (2, 3), (0, 2), or
(3, 2) depending on which corner the patch sits in). With one face +
one place, total budget is around 5-6 ticks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.constants import Action, BlockType, Direction
from factoriax.state import EnvParams, EnvState

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


def place_miner_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Walk adjacent to an ore tile, face it, then ``PLACE_MINER``.

    Reads ``state.map``, ``state.player_positions``, and
    ``state.player_directions`` directly. If any ore tile is
    immediately adjacent in the player's facing direction, place;
    else if at distance 1 from some ore tile, face toward it; else
    step toward the nearest ore tile.

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
    facing = int(np.asarray(state.player_directions[0]))

    ore_mask = np.isin(map_arr, _ORE_BLOCK_VALUES)
    ys, xs = np.where(ore_mask)
    if xs.size == 0:
        return _action(int(Action.NOOP))

    distances = np.abs(xs - px) + np.abs(ys - py)
    not_self = (xs != px) | (ys != py)
    distances = np.where(not_self, distances, np.iinfo(np.int32).max)
    closest = int(np.argmin(distances))
    target_x, target_y = int(xs[closest]), int(ys[closest])
    dist = int(distances[closest])

    if dist == 1:
        if target_x > px:
            need_dir = int(Direction.RIGHT)
            face_act = int(Action.FACE_RIGHT)
        elif target_x < px:
            need_dir = int(Direction.LEFT)
            face_act = int(Action.FACE_LEFT)
        elif target_y > py:
            need_dir = int(Direction.DOWN)
            face_act = int(Action.FACE_DOWN)
        else:
            need_dir = int(Direction.UP)
            face_act = int(Action.FACE_UP)
        if facing == need_dir:
            return _action(int(Action.PLACE_MINER))
        return _action(face_act)

    # Distance > 1: greedy manhattan step toward target.
    if px < target_x:
        return _action(int(Action.RIGHT))
    if px > target_x:
        return _action(int(Action.LEFT))
    if py < target_y:
        return _action(int(Action.DOWN))
    if py > target_y:
        return _action(int(Action.UP))
    return _action(int(Action.NOOP))
