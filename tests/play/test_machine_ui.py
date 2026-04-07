"""Tests for the render_machine_menu UI function.

Validates output shape/dtype, click-region structure, and that the renderer
handles all machine types, filled/empty inventories, and focused-item state
without crashing.  Visual correctness is verified by running the game.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.constants import (
    MACHINE_INVENTORY_COUNT_DTYPE,
    NUM_ITEM_TYPES,
    ItemType,
    MachineType,
)
from factoriax.play.ui import ClickRegion, render_machine_menu

_SW = 320
_SH = 320


# ---------------------------------------------------------------------------
# Output-contract tests
# ---------------------------------------------------------------------------


class TestRenderMachineMenuShape:
    """render_machine_menu must always return a correctly shaped RGBA array."""

    @pytest.mark.parametrize(
        "machine_type",
        [MachineType.MINER, MachineType.CHEST, MachineType.ASSEMBLER],
    )
    def test_returns_uint8_rgba(self, state_factory, machine_type: MachineType) -> None:
        """Output must be uint8 RGBA matching the requested screen dimensions."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(machine_type), dtype=jnp.int32),
        )
        result, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.dtype == np.uint8
        assert result.shape == (_SH, _SW, 4)
        assert isinstance(regions, list)
        assert all(isinstance(r, ClickRegion) for r in regions)


# ---------------------------------------------------------------------------
# Click-region structure
# ---------------------------------------------------------------------------


class TestMachineMenuClickRegions:
    """Click regions must cover machine items and player item types."""

    def test_empty_machine_has_no_machine_slot_regions(
        self, state_factory,
    ) -> None:
        """Empty machine inventory produces zero select_machine_slot regions."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        slot_regions = [r for r in regions if r.action == "select_machine_slot"]
        assert len(slot_regions) == 0

    def test_populated_machine_has_correct_region_count(
        self, state_factory,
    ) -> None:
        """Machine with 3 item types produces 3 select_machine_slot regions."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[0, 0, int(ItemType.COAL)].set(10)
        machine_inv = machine_inv.at[0, 0, int(ItemType.IRON)].set(5)
        machine_inv = machine_inv.at[0, 0, int(ItemType.COPPER)].set(3)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.ASSEMBLER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        slot_regions = [r for r in regions if r.action == "select_machine_slot"]
        assert len(slot_regions) == 3

    def test_machine_slot_params_are_item_types(self, state_factory) -> None:
        """Slot region params are the item type indices of non-empty items."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[0, 0, int(ItemType.IRON)].set(5)
        machine_inv = machine_inv.at[0, 0, int(ItemType.COPPER)].set(3)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.ASSEMBLER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        slot_regions = [r for r in regions if r.action == "select_machine_slot"]
        assert [r.param for r in slot_regions] == [
            int(ItemType.IRON),
            int(ItemType.COPPER),
        ]

    def test_player_inventory_region_count(self, state_factory) -> None:
        """One select_slot region per non-EMPTY item type."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        inv_regions = [r for r in regions if r.action == "select_slot"]
        assert len(inv_regions) == NUM_ITEM_TYPES - 1

    def test_player_inventory_params_are_item_types(self, state_factory) -> None:
        """Player slot region params are item type indices 1..NUM_ITEM_TYPES-1."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        inv_regions = [r for r in regions if r.action == "select_slot"]
        assert [r.param for r in inv_regions] == list(range(1, NUM_ITEM_TYPES))


# ---------------------------------------------------------------------------
# Slot contents rendering
# ---------------------------------------------------------------------------


class TestMachineMenuContents:
    """Renderer must not crash for any combination of filled/empty inventories."""

    def test_all_empty(self, state_factory) -> None:
        """Empty machine inventory renders cleanly."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)

    def test_populated_miner(self, state_factory) -> None:
        """Miner with coal output renders without crash."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[2, 3, int(ItemType.COAL)].set(12)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 3, 2)
        assert result.shape == (_SH, _SW, 4)

    def test_multiple_item_types(self, state_factory) -> None:
        """Assembler with multiple item types renders cleanly."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[0, 0, int(ItemType.COPPER)].set(5)
        machine_inv = machine_inv.at[0, 0, int(ItemType.IRON)].set(5)
        machine_inv = machine_inv.at[0, 0, int(ItemType.COAL)].set(10)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.ASSEMBLER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)

    def test_full_chest_inventory(self, state_factory) -> None:
        """Chest with many item types filled renders cleanly."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        for i, item in enumerate(
            [ItemType.COAL, ItemType.IRON, ItemType.COPPER, ItemType.MINER],
        ):
            machine_inv = machine_inv.at[0, 0, int(item)].set(i + 1)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.CHEST), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)

    def test_player_inventory_shown(self, state_factory) -> None:
        """Player inventory items appear in the strip (no crash)."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.IRON)].set(8)
        inv = inv.at[0, int(ItemType.COPPER)].set(3)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            player_inventory=inv,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)


# ---------------------------------------------------------------------------
# Focused-item highlighting
# ---------------------------------------------------------------------------


class TestMachineMenuFocusedItem:
    """Focused-item state renders without crash for various item types."""

    @pytest.mark.parametrize(
        "focused_item", [int(ItemType.COAL), int(ItemType.IRON)],
    )
    def test_focused_machine_items(
        self, state_factory, focused_item: int,
    ) -> None:
        """Different focused machine items render without crash."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[0, 0, int(ItemType.COAL)].set(5)
        machine_inv = machine_inv.at[0, 0, int(ItemType.IRON)].set(3)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        result, _ = render_machine_menu(
            state, _SW, _SH, 0, 0, focused_machine_item=focused_item,
        )
        assert result.shape == (_SH, _SW, 4)

    @pytest.mark.parametrize(
        "focused_item",
        [int(ItemType.COAL), int(ItemType.IRON), int(ItemType.COPPER)],
    )
    def test_focused_player_items(
        self, state_factory, focused_item: int,
    ) -> None:
        """Different focused player items render without crash."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.CHEST), dtype=jnp.int32,
            ),
        )
        result, _ = render_machine_menu(
            state, _SW, _SH, 0, 0,
            machine_panel_active=False,
            selected_item=focused_item,
        )
        assert result.shape == (_SH, _SW, 4)


# ---------------------------------------------------------------------------
# Tile coordinate addressing
# ---------------------------------------------------------------------------


class TestMachineMenuTileCoords:
    """Menu reads machine data from the correct tile, not always (0, 0)."""

    def test_non_origin_tile(self, state_factory) -> None:
        """Machine at (2, 3) is correctly inspected."""
        machine_types = jnp.zeros((4, 4), dtype=jnp.int32)
        machine_types = machine_types.at[3, 2].set(int(MachineType.CHEST))
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[3, 2, int(ItemType.COAL)].set(10)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=machine_types,
            machine_inventory=machine_inv,
        )
        _, regions = render_machine_menu(state, _SW, _SH, 2, 3)
        slot_regions = [r for r in regions if r.action == "select_machine_slot"]
        assert len(slot_regions) == 1
        assert slot_regions[0].param == int(ItemType.COAL)

    def test_tile_inventory_isolation(self, state_factory) -> None:
        """Items at tile (1,1) are not shown when inspecting tile (0,0)."""
        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[1, 1, int(ItemType.COAL)].set(99)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        # Inspecting (0, 0): should have no machine slot regions (empty).
        result_00, regions_00 = render_machine_menu(state, _SW, _SH, 0, 0)
        # Inspecting (1, 1): should have 1 machine slot region (coal).
        result_11, regions_11 = render_machine_menu(state, _SW, _SH, 1, 1)
        slots_00 = [r for r in regions_00 if r.action == "select_machine_slot"]
        slots_11 = [r for r in regions_11 if r.action == "select_machine_slot"]
        assert len(slots_00) == 0
        assert len(slots_11) == 1
        assert result_00.shape == (_SH, _SW, 4)
        assert result_11.shape == (_SH, _SW, 4)
        assert not np.array_equal(result_00, result_11)
