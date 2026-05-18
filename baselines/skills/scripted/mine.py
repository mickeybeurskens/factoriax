"""Scripted baseline for the L.2 mine skill.

Mining is look-at: the player must stand adjacent to an ore tile and
face it. The policy navigates to a tile adjacent to the nearest ore,
issues ``FACE_*`` if needed, and ``MINE`` once aligned. If the agent
has wandered onto an ore tile (it is walkable), it steps off first so
it can re-approach from an adjacent square. NOOP only if the map has
no ore at all (degenerate case the level builder shouldn't produce).

Reads the terrain map and inventory directly from :class:`EnvState`.
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

# (dx, dy) per Direction value.
_DIR_OFFSETS: dict[int, tuple[int, int]] = {
    int(Direction.LEFT): (-1, 0),
    int(Direction.RIGHT): (1, 0),
    int(Direction.UP): (0, -1),
    int(Direction.DOWN): (0, 1),
}

# Blocked terrain (ignored as adjacent slots and step-off targets).
_BLOCKED_BLOCK_VALUES: tuple[int, ...] = (
    int(BlockType.WATER),
    int(BlockType.OUT_OF_BOUNDS),
)


def _action(a: int) -> jax.Array:
    """Wrap an int as a JAX int32 scalar (mypy-friendly)."""
    out: jax.Array = jnp.asarray(a, dtype=jnp.int32)
    return out


def _face_action_toward(dx: int, dy: int) -> int:
    """Return the FACE_* action that points toward ``(dx, dy)``."""
    if dx > 0:
        return int(Action.FACE_RIGHT)
    if dx < 0:
        return int(Action.FACE_LEFT)
    if dy > 0:
        return int(Action.FACE_DOWN)
    return int(Action.FACE_UP)


def _move_action_toward(dx: int, dy: int) -> int:
    """Return the movement action that advances toward ``(dx, dy)``."""
    if dx > 0:
        return int(Action.RIGHT)
    if dx < 0:
        return int(Action.LEFT)
    if dy > 0:
        return int(Action.DOWN)
    return int(Action.UP)


def mine_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Face the nearest ore from an adjacent tile, then ``MINE``.

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
    direction = int(state.player_directions[0])
    px, py = int(pos[0]), int(pos[1])
    h, w = map_arr.shape

    # If the tile in front is ore, mine it.
    dx, dy = _DIR_OFFSETS.get(direction, (0, 0))
    fx, fy = px + dx, py + dy
    if 0 <= fx < w and 0 <= fy < h and int(map_arr[fy, fx]) in _ORE_BLOCK_VALUES:
        return _action(int(Action.MINE))

    ys, xs = np.where(np.isin(map_arr, _ORE_BLOCK_VALUES))
    if xs.size == 0:
        return _action(int(Action.NOOP))

    distances = np.abs(xs - px) + np.abs(ys - py)
    closest = int(np.argmin(distances))
    target_x, target_y = int(xs[closest]), int(ys[closest])

    # Adjacent to the nearest ore: face it for next-tick mine.
    if abs(target_x - px) + abs(target_y - py) == 1:
        return _action(_face_action_toward(target_x - px, target_y - py))

    # Standing on an ore (it is walkable): step off so we can approach
    # from a neighbour and face it.
    if int(map_arr[py, px]) in _ORE_BLOCK_VALUES:
        for ndx, ndy, act in (
            (1, 0, int(Action.RIGHT)),
            (-1, 0, int(Action.LEFT)),
            (0, 1, int(Action.DOWN)),
            (0, -1, int(Action.UP)),
        ):
            nx, ny = px + ndx, py + ndy
            if 0 <= nx < w and 0 <= ny < h:
                block = int(map_arr[ny, nx])
                if block in _ORE_BLOCK_VALUES or block in _BLOCKED_BLOCK_VALUES:
                    continue
                return _action(act)
        return _action(int(Action.NOOP))

    # Navigate greedily toward the ore tile.
    return _action(_move_action_toward(target_x - px, target_y - py))
