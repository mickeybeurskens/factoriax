"""Integration test: editor machine inventories survive into EnvState.

When the editor places a miner with coal in its inventory and launches
play mode, the coal should end up in the entity's fuel field so the
miner starts working immediately. This catches the mismatch where
build_state ignores Level.machine_inventory.
"""

from __future__ import annotations

import numpy as np

from factoriax.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.levels import Level, build_state
from factoriax.state import EnvParams


def _make_level_with_fueled_miner() -> Level:
    """Create a Level with a miner that has coal in its inventory."""
    h, w = 5, 5
    block_map = np.full((h, w), int(BlockType.DIRT), dtype=np.int32)
    block_map[2, 2] = int(BlockType.IRON)

    resources = np.zeros((h, w), dtype=np.int32)
    resources[2, 2] = 100

    machine_types = np.full(
        (h, w), int(MachineType.NONE), dtype=np.int32,
    )
    machine_types[2, 2] = int(MachineType.MINER)

    machine_dirs = np.zeros((h, w), dtype=np.int32)
    machine_dirs[2, 2] = int(Direction.DOWN)

    machine_inv = np.zeros((h, w, NUM_ITEM_TYPES), dtype=np.int32)
    machine_inv[2, 2, int(ItemType.COAL)] = 5

    return Level(
        name="test_fueled",
        map_width=w,
        map_height=h,
        block_map=block_map,
        block_resources=resources,
        machine_types=machine_types,
        machine_directions=machine_dirs,
        machine_inventory=machine_inv,
        player_positions=[(1, 1)],
    )


def _make_level_with_pallet_items() -> Level:
    """Create a Level with a pallet containing iron ore."""
    h, w = 5, 5
    block_map = np.full((h, w), int(BlockType.DIRT), dtype=np.int32)

    machine_types = np.full(
        (h, w), int(MachineType.NONE), dtype=np.int32,
    )
    machine_types[2, 2] = int(MachineType.PALLET)

    machine_inv = np.zeros((h, w, NUM_ITEM_TYPES), dtype=np.int32)
    machine_inv[2, 2, int(ItemType.IRON_ORE)] = 10

    return Level(
        name="test_pallet",
        map_width=w,
        map_height=h,
        block_map=block_map,
        machine_types=machine_types,
        machine_inventory=machine_inv,
        player_positions=[(0, 0)],
    )


class TestEditorInventoryInit:
    """Machine inventories from the editor must appear in EnvState."""

    def test_miner_coal_goes_to_fuel(self) -> None:
        """Coal in a miner's inventory should populate ent_fuel."""
        level = _make_level_with_fueled_miner()
        params = EnvParams(map_width=5, map_height=5, num_players=1)
        state = build_state(level, params)

        eidx = int(state.tile_entity[2, 2])
        assert eidx >= 0, "Miner entity not found"
        assert int(state.ent_fuel[eidx]) == 5, (
            f"Expected fuel=5, got {int(state.ent_fuel[eidx])}"
        )

    def test_pallet_items_go_to_buffer(self) -> None:
        """Items in a pallet's inventory should populate buffer."""
        level = _make_level_with_pallet_items()
        params = EnvParams(map_width=5, map_height=5, num_players=1)
        state = build_state(level, params)

        eidx = int(state.tile_entity[2, 2])
        assert eidx >= 0, "Pallet entity not found"
        assert int(state.ent_buf_type[eidx]) == int(
            ItemType.IRON_ORE,
        ), f"Expected buf_type=IRON_ORE, got {int(state.ent_buf_type[eidx])}"
        assert int(state.ent_buf_count[eidx]) == 10, (
            f"Expected buf_count=10, got {int(state.ent_buf_count[eidx])}"
        )

    def test_miner_without_inventory_has_zero_fuel(self) -> None:
        """A miner with no inventory should have fuel=0."""
        h, w = 3, 3
        block_map = np.full((h, w), int(BlockType.IRON), dtype=np.int32)
        machine_types = np.full(
            (h, w), int(MachineType.NONE), dtype=np.int32,
        )
        machine_types[1, 1] = int(MachineType.MINER)

        level = Level(
            name="no_inv",
            map_width=w,
            map_height=h,
            block_map=block_map,
            machine_types=machine_types,
            player_positions=[(0, 0)],
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        state = build_state(level, params)

        eidx = int(state.tile_entity[1, 1])
        assert int(state.ent_fuel[eidx]) == 0
