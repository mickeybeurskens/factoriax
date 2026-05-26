"""Scripted baseline for the L.4 place_miner skill.

The level holds three non-adjacent ore patches; the player spawns at
the centre with 5 miners in inventory. The policy walks toward the
nearest *un-occupied* ore tile (one that doesn't yet have a placed
machine on it), stops one tile short, faces the tile, and emits
``PLACE_MINER``. After placement, the policy retargets the next
un-occupied ore tile until all three patches have a miner — the
achievement (``count_miners_on_ore >= 3``) latches when the third
miner lands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import Action, BlockType, Direction
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
    machine_types = np.asarray(state.machine_types)
    pos = np.asarray(state.player_positions[0])
    px, py = int(pos[0]), int(pos[1])
    facing = int(np.asarray(state.player_directions[0]))

    # Targets are mineable tiles that don't yet have any machine on
    # them — those are the ones still missing a miner.
    ore_mask = np.isin(map_arr, _ORE_BLOCK_VALUES)
    no_machine_mask = machine_types == 0
    target_mask = ore_mask & no_machine_mask
    ys, xs = np.where(target_mask)
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
