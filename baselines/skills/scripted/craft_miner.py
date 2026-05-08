"""Scripted baseline for the L.3 craft_miner skill.

The level pre-places two pallets at non-adjacent positions; one holds
``IRON_PLATE``, the other ``WIRE``. The policy walks toward the
nearest still-loaded pallet (any pallet whose buffer is non-empty),
faces it, ``WITHDRAW``s the contents, then targets the next loaded
pallet. Once both ingredients are in player inventory, it emits
``CRAFT_MINER``.

Reads ``state.machine_types``, ``state.ent_buf_count``,
``state.ent_buf_type``, ``state.tile_entity``, ``state.player_*``
directly. Scripted baselines are state-readers, not obs-readers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.constants import Action, Direction, ItemType, MachineType
from factoriax.state import EnvParams, EnvState


def _action(a: int) -> jax.Array:
    """Wrap an int as a JAX int32 scalar."""
    out: jax.Array = jnp.asarray(a, dtype=jnp.int32)
    return out


def _has_ingredients(state: EnvState) -> bool:
    """True when player 0 holds both IRON_PLATE and WIRE."""
    inv = np.asarray(state.player_inventory[0])
    return bool(inv[int(ItemType.IRON_PLATE)] >= 1 and inv[int(ItemType.WIRE)] >= 1)


def _loaded_pallet_tiles(state: EnvState) -> list[tuple[int, int]]:
    """Return ``(x, y)`` for every pallet tile whose buffer holds an item."""
    machines = np.asarray(state.machine_types)
    tile_entity = np.asarray(state.tile_entity)
    buf_counts = np.asarray(state.ent_buf_count)
    ys, xs = np.where(machines == int(MachineType.PALLET))
    loaded: list[tuple[int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        eidx = int(tile_entity[y, x])
        if eidx >= 0 and int(buf_counts[eidx]) > 0:
            loaded.append((int(x), int(y)))
    return loaded


def craft_miner_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Withdraw ingredients from two pallets, then ``CRAFT_MINER``.

    Args:
        state: Current environment state.
        params: Environment parameters (unused).

    Returns:
        JAX int32 scalar action.
    """
    del params

    if _has_ingredients(state):
        return _action(int(Action.CRAFT_MINER))

    targets = _loaded_pallet_tiles(state)
    if not targets:
        # No loaded pallets but ingredients still missing — unsolvable
        # state, fall back to NOOP. Shouldn't happen on canonical seeds.
        return _action(int(Action.NOOP))

    pos = np.asarray(state.player_positions[0])
    px, py = int(pos[0]), int(pos[1])
    facing = int(np.asarray(state.player_directions[0]))

    distances = [abs(tx - px) + abs(ty - py) for tx, ty in targets]
    closest_idx = int(np.argmin(distances))
    target_x, target_y = targets[closest_idx]
    dist = distances[closest_idx]

    if dist == 1:
        if target_x > px:
            need_dir, face_act = int(Direction.RIGHT), int(Action.FACE_RIGHT)
        elif target_x < px:
            need_dir, face_act = int(Direction.LEFT), int(Action.FACE_LEFT)
        elif target_y > py:
            need_dir, face_act = int(Direction.DOWN), int(Action.FACE_DOWN)
        else:
            need_dir, face_act = int(Direction.UP), int(Action.FACE_UP)
        if facing == need_dir:
            return _action(int(Action.WITHDRAW))
        return _action(face_act)

    if px < target_x:
        return _action(int(Action.RIGHT))
    if px > target_x:
        return _action(int(Action.LEFT))
    if py < target_y:
        return _action(int(Action.DOWN))
    if py > target_y:
        return _action(int(Action.UP))
    return _action(int(Action.NOOP))
