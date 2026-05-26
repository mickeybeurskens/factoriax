"""Integration test: editor machine inventories survive into EnvState.

When the editor places a pallet with items in its inventory and
launches play mode, those items should end up in the entity's buffer
so the factory starts in the expected state. This catches mismatches
where build_state ignores Level.machine_inventory.
"""

from __future__ import annotations

import numpy as np

from factoriax.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.levels import Level, build_state
from factoriax.state import EnvParams


def _make_level_with_pallet_items() -> Level:
    """Create a Level with a pallet containing iron ore."""
    h, w = 5, 5
    block_map = np.full((h, w), int(BlockType.DIRT), dtype=np.int32)

    machine_types = np.full(
        (h, w),
        int(Machine.NONE),
        dtype=np.int32,
    )
    machine_types[2, 2] = int(Machine.PALLET)

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
