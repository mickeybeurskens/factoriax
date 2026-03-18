"""Tests for the render_machine_menu UI function.

Validates output shape/dtype, click-region structure, and that the renderer
handles all machine types, filled/empty slots, and focused-slot state without
crashing.  Visual correctness is verified by running the game.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    NUM_INVENTORY_SLOTS,
    MACHINE_NUM_SLOTS,
    MACHINE_SLOT_ROLES,
    SLOT_ROLE_COLORS,
    SLOT_ROLE_LABELS,
    ItemType,
    MachineType,
    SlotRole,
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
    """Click regions must cover every active machine slot and every player slot."""

    @pytest.mark.parametrize(
        "machine_type,expected_slots",
        [
            (MachineType.MINER, int(MACHINE_NUM_SLOTS[MachineType.MINER])),
            (MachineType.CHEST, int(MACHINE_NUM_SLOTS[MachineType.CHEST])),
            (MachineType.ASSEMBLER, int(MACHINE_NUM_SLOTS[MachineType.ASSEMBLER])),
        ],
    )
    def test_machine_slot_region_count(
        self, state_factory, machine_type: MachineType, expected_slots: int
    ) -> None:
        """One select_machine_slot region per active slot, no more, no less."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(machine_type), dtype=jnp.int32),
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        slot_regions = [r for r in regions if r.action == "select_machine_slot"]
        assert len(slot_regions) == expected_slots

    def test_machine_slot_params_are_sequential(self, state_factory) -> None:
        """Slot region params must be 0, 1, ..., N-1 in order."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.ASSEMBLER), dtype=jnp.int32
            ),
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        slot_regions = [r for r in regions if r.action == "select_machine_slot"]
        expected = list(range(int(MACHINE_NUM_SLOTS[MachineType.ASSEMBLER])))
        assert [r.param for r in slot_regions] == expected

    def test_player_inventory_region_count(self, state_factory) -> None:
        """One select_slot region per player inventory slot."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.MINER), dtype=jnp.int32),
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        inv_regions = [r for r in regions if r.action == "select_slot"]
        assert len(inv_regions) == NUM_INVENTORY_SLOTS

    def test_player_inventory_params_are_sequential(self, state_factory) -> None:
        """Player slot region params must be 0, 1, ..., NUM_INVENTORY_SLOTS-1."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.MINER), dtype=jnp.int32),
        )
        _, regions = render_machine_menu(state, _SW, _SH, 0, 0)
        inv_regions = [r for r in regions if r.action == "select_slot"]
        assert [r.param for r in inv_regions] == list(range(NUM_INVENTORY_SLOTS))


# ---------------------------------------------------------------------------
# Slot contents rendering
# ---------------------------------------------------------------------------


class TestMachineMenuContents:
    """Renderer must not crash for any combination of filled/empty slots."""

    def test_all_empty_slots(self, state_factory) -> None:
        """All-empty machine inventory renders cleanly."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.MINER), dtype=jnp.int32),
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)

    def test_populated_output_slot(self, state_factory) -> None:
        """Miner output slot (index 1) with coal renders without crash."""
        inv_items = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[2, 3, 1].set(int(ItemType.COAL))
        inv_counts = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
        inv_counts = inv_counts.at[2, 3, 1].set(12)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.MINER), dtype=jnp.int32),
            machine_inventory_items=inv_items,
            machine_inventory_counts=inv_counts,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 3, 2)
        assert result.shape == (_SH, _SW, 4)

    def test_multiple_input_slots_populated(self, state_factory) -> None:
        """Assembler with all 3 input slots filled renders cleanly."""
        inv_items = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0, 0].set(int(ItemType.COPPER))
        inv_items = inv_items.at[0, 0, 1].set(int(ItemType.IRON))
        inv_items = inv_items.at[0, 0, 2].set(int(ItemType.COAL))
        inv_counts = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
        inv_counts = inv_counts.at[0, 0, 0].set(5)
        inv_counts = inv_counts.at[0, 0, 1].set(5)
        inv_counts = inv_counts.at[0, 0, 2].set(10)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.ASSEMBLER), dtype=jnp.int32
            ),
            machine_inventory_items=inv_items,
            machine_inventory_counts=inv_counts,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)

    def test_full_chest_inventory(self, state_factory) -> None:
        """Chest with all 8 slots filled (mixed items) renders cleanly."""
        items = [
            ItemType.COAL, ItemType.IRON, ItemType.COPPER, ItemType.MINER,
            ItemType.COAL, ItemType.IRON, ItemType.COPPER, ItemType.MINER,
        ]
        inv_items = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
        for i, item in enumerate(items):
            inv_items = inv_items.at[0, 0, i].set(int(item))
            inv_counts = inv_counts.at[0, 0, i].set(i + 1)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.CHEST), dtype=jnp.int32),
            machine_inventory_items=inv_items,
            machine_inventory_counts=inv_counts,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)

    def test_player_inventory_shown(self, state_factory) -> None:
        """Player inventory items appear in the strip (no crash)."""
        inv_items = jnp.array(
            [[ItemType.IRON, ItemType.COPPER, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=jnp.int32,
        )
        inv_counts = jnp.array(
            [[8, 3, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=jnp.int32,
        )
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.MINER), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)


# ---------------------------------------------------------------------------
# Focused-slot highlighting
# ---------------------------------------------------------------------------


class TestMachineMenuFocusedSlot:
    """Focused-slot state renders without crash for any valid slot index."""

    @pytest.mark.parametrize("focused", [0, 1])
    def test_miner_focused_slots(self, state_factory, focused: int) -> None:
        """Both miner slots can be focused without crash."""
        sel = jnp.zeros((4, 4), dtype=jnp.int32)
        sel = sel.at[0, 0].set(focused)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.MINER), dtype=jnp.int32),
            machine_selected_slot=sel,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)

    @pytest.mark.parametrize("focused", range(int(MACHINE_NUM_SLOTS[MachineType.CHEST])))
    def test_chest_all_focused_slots(self, state_factory, focused: int) -> None:
        """Each of the 8 chest slots can be focused without crash."""
        sel = jnp.zeros((4, 4), dtype=jnp.int32)
        sel = sel.at[0, 0].set(focused)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.CHEST), dtype=jnp.int32),
            machine_selected_slot=sel,
        )
        result, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        assert result.shape == (_SH, _SW, 4)


# ---------------------------------------------------------------------------
# Tile coordinate addressing
# ---------------------------------------------------------------------------


class TestMachineMenuTileCoords:
    """Menu reads machine data from the correct tile, not always (0, 0)."""

    def test_non_origin_tile(self, state_factory) -> None:
        """Machine at (2, 3) is correctly inspected and yields chest slot count."""
        machine_types = jnp.zeros((4, 4), dtype=jnp.int32)
        machine_types = machine_types.at[3, 2].set(int(MachineType.CHEST))
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=machine_types,
        )
        _, regions = render_machine_menu(state, _SW, _SH, 2, 3)
        slot_regions = [r for r in regions if r.action == "select_machine_slot"]
        assert len(slot_regions) == int(MACHINE_NUM_SLOTS[MachineType.CHEST])

    def test_tile_inventory_isolation(self, state_factory) -> None:
        """Items at tile (1,1) are not shown when inspecting tile (0,0)."""
        inv_items = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[1, 1, 1].set(int(ItemType.COAL))
        inv_counts = jnp.zeros((4, 4, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
        inv_counts = inv_counts.at[1, 1, 1].set(99)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.MINER), dtype=jnp.int32),
            machine_inventory_items=inv_items,
            machine_inventory_counts=inv_counts,
        )
        # Inspecting (0, 0): output slot should be empty
        result_00, _ = render_machine_menu(state, _SW, _SH, 0, 0)
        result_11, _ = render_machine_menu(state, _SW, _SH, 1, 1)
        # Both render cleanly; they produce different pixels
        assert result_00.shape == (_SH, _SW, 4)
        assert result_11.shape == (_SH, _SW, 4)
        assert not np.array_equal(result_00, result_11)
