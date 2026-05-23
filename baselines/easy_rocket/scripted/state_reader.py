"""Pure-function helpers that read :class:`EnvState`.

The scripted agent and layout planner operate Python-side on plain
ints, tuples, and dataclasses. These helpers convert JAX array
fields of :class:`EnvState` at the boundary and return numpy /
Python values. None of them emit actions or mutate state.

Coordinate convention follows the engine: tile positions are
``(x, y)`` with ``x`` the column and ``y`` the row. ``state.map``
and the other grids are indexed ``[y, x]``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from factoriax.constants import BlockType, MachineType
from factoriax.state import EnvState

# Block types that hold mineable ore. Order matches the easy_rocket
# level builder's ``_PATCH_BLOCKS`` tuple, but the planner uses the
# set, not the order.
_ORE_BLOCKS: tuple[int, ...] = (
    int(BlockType.IRON),
    int(BlockType.COPPER),
    int(BlockType.TIN),
    int(BlockType.SILICON),
    int(BlockType.COAL),
    int(BlockType.LIMESTONE),
)

# Block types the player can walk onto without a machine in the way.
# Everything else (ore, water, the out-of-bounds boundary) is
# impassable as terrain.
_WALKABLE_BLOCKS: frozenset[int] = frozenset(
    {
        int(BlockType.INVALID),
        int(BlockType.DIRT),
    }
)


@dataclasses.dataclass(frozen=True)
class OrePatch:
    """A group of tiles sharing one ore block type with resources left.

    Attributes:
        ore_block: ``BlockType`` integer of the patch's ore.
        tiles: Tile coordinates as ``(x, y)`` tuples. For easy_rocket
            this is exactly the 2x2 footprint placed by the level
            builder. Tiles with depleted ``block_resources`` are
            excluded.
    """

    ore_block: int
    tiles: tuple[tuple[int, int], ...]


def find_patches(state: EnvState) -> list[OrePatch]:
    """Locate every ore patch on the current map.

    Returns one :class:`OrePatch` per ore block type present, grouping
    all tiles of that type into a single patch. Sufficient for
    easy_rocket's one-patch-per-ore-type level builder; for hypothetical
    levels with multiple disjoint patches of the same ore, this would
    need a connected-components pass.

    Args:
        state: Current environment state.

    Returns:
        List of patches in the order ore blocks first appear in
        :data:`_ORE_BLOCKS`. Empty list if the map holds no ore.
    """
    map_arr = np.asarray(state.map)
    resources = np.asarray(state.block_resources)

    patches: list[OrePatch] = []
    for ore in _ORE_BLOCKS:
        mask = (map_arr == ore) & (resources > 0)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        tiles = tuple((int(x), int(y)) for x, y in zip(xs, ys, strict=True))
        patches.append(OrePatch(ore_block=ore, tiles=tiles))
    return patches


def player_pos(state: EnvState, player: int = 0) -> tuple[int, int]:
    """Return ``player``'s tile position as ``(x, y)``."""
    pos = np.asarray(state.player_positions[player])
    return int(pos[0]), int(pos[1])


def player_direction(state: EnvState, player: int = 0) -> int:
    """Return ``player``'s facing as a ``Direction`` integer."""
    return int(np.asarray(state.player_directions[player]))


def inv_count(state: EnvState, item: int, player: int = 0) -> int:
    """Return how many of ``item`` ``player`` holds in their inventory."""
    inv = np.asarray(state.player_inventory[player])
    return int(inv[item])


def block_at(state: EnvState, x: int, y: int) -> int:
    """Return the ``BlockType`` integer at tile ``(x, y)``."""
    return int(np.asarray(state.map)[y, x])


def machine_at(state: EnvState, x: int, y: int) -> int:
    """Return the ``MachineType`` integer at ``(x, y)``; 0 means none."""
    return int(np.asarray(state.machine_types)[y, x])


def entity_at(state: EnvState, x: int, y: int) -> int:
    """Return the entity index at ``(x, y)``; ``-1`` if no entity."""
    return int(np.asarray(state.tile_entity)[y, x])


def tile_free(state: EnvState, x: int, y: int) -> bool:
    """Return True if ``(x, y)`` is walkable and unoccupied by a machine.

    Walkable means the underlying block type is in
    :data:`_WALKABLE_BLOCKS` (DIRT or INVALID at start) and no machine
    occupies the tile. Ore tiles, water, and out-of-bounds are not
    free. The planner uses this to validate every candidate placement
    site before committing it.
    """
    map_arr = np.asarray(state.map)
    h, w = map_arr.shape
    if not (0 <= x < w and 0 <= y < h):
        return False
    if int(map_arr[y, x]) not in _WALKABLE_BLOCKS:
        return False
    if int(np.asarray(state.machine_types)[y, x]) != int(MachineType.NONE):
        return False
    return True
