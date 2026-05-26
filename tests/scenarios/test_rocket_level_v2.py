"""Tests for the v2 streamlined rocket scenario map.

The v2 layout is:

- A 1-tile-wide coal column on ``x = 0``, full map height.
- Five 2x2 ore patches at ``(3, 9), (3, 12), (3, 15), (3, 18), (3, 21)``
  for iron / copper / tin / silicon / limestone respectively.
- Spawn at the centre, with a furnace immediately west and an
  assembler immediately east.

These tests are pure-Python (no env rollout) and run in milliseconds.
They guard the geometry the M3+ agent rewrites depend on.
"""

from __future__ import annotations

import numpy as np

from factoriax.constants import BlockType, Direction, Machine
from factoriax.scenarios import build_rocket_level

_MAP_SIZE = 32
_PATCH_SIZE = 2
_PATCHES: list[tuple[int, int, BlockType]] = [
    (3, 9, BlockType.IRON),
    (3, 12, BlockType.COPPER),
    (3, 15, BlockType.TIN),
    (3, 18, BlockType.SILICON),
    (3, 21, BlockType.LIMESTONE),
]


def test_coal_column_fills_full_left_edge() -> None:
    """Every tile on x=0 carries BlockType.COAL with non-zero resources."""
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    resources = np.asarray(level.block_resources)
    for y in range(_MAP_SIZE):
        assert int(block_map[y, 0]) == int(BlockType.COAL), f"(0, {y}) is not COAL"
        assert int(resources[y, 0]) > 0, f"(0, {y}) has zero coal resource"


def test_ore_patches_at_expected_v2_coords() -> None:
    """Each 2x2 patch lands on cols 3-4 at the documented top-left row."""
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    for top_x, top_y, block in _PATCHES:
        for dx in range(_PATCH_SIZE):
            for dy in range(_PATCH_SIZE):
                tile = (top_x + dx, top_y + dy)
                assert int(block_map[tile[1], tile[0]]) == int(block), (
                    f"{block.name} patch missing at {tile}"
                )


def test_dirt_buffer_between_coal_and_ores() -> None:
    """Cols 1-2 across every patch row are dirt — the 2-tile gap."""
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    for _, top_y, _ in _PATCHES:
        for dy in range(_PATCH_SIZE):
            for x in (1, 2):
                tile = (x, top_y + dy)
                assert int(block_map[tile[1], tile[0]]) == int(BlockType.DIRT), (
                    f"{tile} should be DIRT, got "
                    f"{BlockType(int(block_map[tile[1], tile[0]])).name}"
                )


def test_pre_placed_furnace_and_assembler_unchanged() -> None:
    """Furnace at (15, 16), assembler at (17, 16) — same as v1."""
    level = build_rocket_level()
    machine_types = np.asarray(level.machine_types)
    machine_directions = np.asarray(level.machine_directions)
    assert int(machine_types[16, 15]) == int(Machine.FURNACE)
    assert int(machine_types[16, 17]) == int(Machine.ASSEMBLER)
    assert int(machine_directions[16, 15]) == int(Direction.DOWN)
    assert int(machine_directions[16, 17]) == int(Direction.DOWN)


def test_player_spawns_at_centre() -> None:
    """Spawn stays at (16, 16) so existing skills' navigation works."""
    level = build_rocket_level()
    spawn = np.asarray(level.player_positions)
    assert spawn.shape[0] >= 1
    assert int(spawn[0, 0]) == 16
    assert int(spawn[0, 1]) == 16


def test_factory_zone_east_of_patches_is_dirt() -> None:
    """Cols 5-31, every row, are all dirt — the factory build area.

    Only exception: row 16, cols 15+17 (pre-placed F+A).
    """
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    machine_types = np.asarray(level.machine_types)
    for y in range(_MAP_SIZE):
        for x in range(5, _MAP_SIZE):
            block = int(block_map[y, x])
            mt = int(machine_types[y, x])
            assert block == int(BlockType.DIRT), (
                f"({x}, {y}) factory zone should be DIRT, got "
                f"{BlockType(block).name} (machine={Machine(mt).name})"
            )


def test_no_overlap_between_coal_column_and_ore_patches() -> None:
    """The 2-tile dirt gap at cols 1-2 separates them — sanity check."""
    coal_tiles: set[tuple[int, int]] = {(0, y) for y in range(_MAP_SIZE)}
    ore_tiles: set[tuple[int, int]] = set()
    for top_x, top_y, _ in _PATCHES:
        for dx in range(_PATCH_SIZE):
            for dy in range(_PATCH_SIZE):
                ore_tiles.add((top_x + dx, top_y + dy))
    assert coal_tiles.isdisjoint(ore_tiles)
