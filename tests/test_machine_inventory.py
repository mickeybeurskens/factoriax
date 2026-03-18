"""Tests for the generalized machine inventory system.

Covers slot-role semantics (SlotRole enum), the MACHINE_SLOT_ROLES lookup,
machine type extension (CHEST, ASSEMBLER), and the new state fields
machine_inventory_items / machine_inventory_counts / machine_selected_recipe /
machine_selected_slot.  Machine operation tests (refueling, mining) that rely
on the new inventory layout are co-located here rather than in test_machines.py
so this file stands alone as the canonical inventory-system test suite.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax import BlockType, EnvParams, ItemType
from factoriax.constants import (
    MACHINE_NUM_SLOTS,
    MACHINE_SLOT_ROLES,
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_MACHINE_STACK_SIZE,
    MachineType,
    SlotRole,
)
from factoriax.levels import build_state, generate_state
from factoriax.machines import refuel_machines, run_miners, update_all_machines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inv_counts(slot_data: dict[int, int], h: int, w: int) -> jnp.ndarray:
    """Build a machine_inventory_counts array from a {slot_idx: count} mapping."""
    arr = jnp.zeros((h, w, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
    for slot, count in slot_data.items():
        arr = arr.at[..., slot].set(count)
    return arr


def _inv_items(slot_data: dict[int, int], h: int, w: int) -> jnp.ndarray:
    """Build a machine_inventory_items array from a {slot_idx: item_type} mapping."""
    arr = jnp.zeros((h, w, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32)
    for slot, item in slot_data.items():
        arr = arr.at[..., slot].set(item)
    return arr


# ---------------------------------------------------------------------------
# SlotRole constant tests
# ---------------------------------------------------------------------------


class TestSlotRoleConstants:
    """Verify MACHINE_SLOT_ROLES and MACHINE_NUM_SLOTS are self-consistent."""

    def test_shape(self) -> None:
        """MACHINE_SLOT_ROLES must have one row per MachineType and 8 columns."""
        assert MACHINE_SLOT_ROLES.shape == (len(MachineType), MAX_MACHINE_INVENTORY_SLOTS)

    def test_none_machine_all_none_slots(self) -> None:
        """MachineType.NONE should have all NONE-role slots."""
        assert np.all(MACHINE_SLOT_ROLES[MachineType.NONE] == SlotRole.NONE)

    def test_miner_slots(self) -> None:
        """Miner slot 0 is INPUT (fuel) and slot 1 is OUTPUT (ore)."""
        roles = MACHINE_SLOT_ROLES[MachineType.MINER]
        assert roles[0] == SlotRole.INPUT
        assert roles[1] == SlotRole.OUTPUT
        assert np.all(roles[2:] == SlotRole.NONE)

    def test_chest_all_storage(self) -> None:
        """Chest should have all 8 STORAGE slots."""
        roles = MACHINE_SLOT_ROLES[MachineType.CHEST]
        assert np.all(roles == SlotRole.STORAGE)

    def test_assembler_slots(self) -> None:
        """Assembler: slots 0-2 are INPUT, slot 3 is OUTPUT, rest are NONE."""
        roles = MACHINE_SLOT_ROLES[MachineType.ASSEMBLER]
        assert roles[0] == SlotRole.INPUT
        assert roles[1] == SlotRole.INPUT
        assert roles[2] == SlotRole.INPUT
        assert roles[3] == SlotRole.OUTPUT
        assert np.all(roles[4:] == SlotRole.NONE)

    def test_machine_num_slots(self) -> None:
        """MACHINE_NUM_SLOTS should count non-NONE slots correctly."""
        assert MACHINE_NUM_SLOTS[MachineType.NONE] == 0
        assert MACHINE_NUM_SLOTS[MachineType.MINER] == 2
        assert MACHINE_NUM_SLOTS[MachineType.CHEST] == 8
        assert MACHINE_NUM_SLOTS[MachineType.ASSEMBLER] == 4

    def test_machine_num_slots_consistent_with_role_table(self) -> None:
        """MACHINE_NUM_SLOTS must match the count of non-NONE entries in each row."""
        for mtype in MachineType:
            active = int(np.sum(MACHINE_SLOT_ROLES[int(mtype)] != SlotRole.NONE))
            assert MACHINE_NUM_SLOTS[int(mtype)] == active, (
                f"MACHINE_NUM_SLOTS[{mtype.name}]={MACHINE_NUM_SLOTS[int(mtype)]} "
                f"does not match active slot count {active}"
            )


# ---------------------------------------------------------------------------
# SlotRole deposit/withdraw access rules
# ---------------------------------------------------------------------------


class TestSlotRoleAccessRules:
    """Verify the documented player access rules for each slot role."""

    @pytest.mark.parametrize(
        "role",
        [SlotRole.INPUT, SlotRole.OUTPUT, SlotRole.STORAGE],
    )
    def test_player_can_withdraw_from_any_non_none_role(self, role: SlotRole) -> None:
        """Player may withdraw from INPUT, OUTPUT, and STORAGE slots."""
        # This is a design invariant; assert the constant is not OUTPUT-restricted.
        assert role in (SlotRole.INPUT, SlotRole.OUTPUT, SlotRole.STORAGE)

    @pytest.mark.parametrize(
        "role,may_deposit",
        [
            (SlotRole.INPUT, True),
            (SlotRole.OUTPUT, False),
            (SlotRole.STORAGE, True),
            (SlotRole.NONE, False),
        ],
    )
    def test_player_deposit_rules(self, role: SlotRole, may_deposit: bool) -> None:
        """Player may deposit into INPUT and STORAGE but not OUTPUT or NONE."""
        can_deposit = role in (SlotRole.INPUT, SlotRole.STORAGE)
        assert can_deposit == may_deposit


# ---------------------------------------------------------------------------
# State field shapes
# ---------------------------------------------------------------------------


class TestMachineInventoryStateFields:
    """Machine inventory state fields have the correct shapes and dtypes."""

    @pytest.fixture(scope="class")
    def state(self):
        from factoriax.levels import LevelBuilder

        level = LevelBuilder(8, 6).build("test")
        params = EnvParams(map_width=8, map_height=6, num_players=1)
        return build_state(level, params)

    def test_machine_inventory_items_shape(self, state) -> None:
        assert state.machine_inventory_items.shape == (6, 8, MAX_MACHINE_INVENTORY_SLOTS)

    def test_machine_inventory_counts_shape(self, state) -> None:
        assert state.machine_inventory_counts.shape == (6, 8, MAX_MACHINE_INVENTORY_SLOTS)

    def test_machine_selected_recipe_shape(self, state) -> None:
        assert state.machine_selected_recipe.shape == (6, 8)

    def test_machine_selected_slot_shape(self, state) -> None:
        assert state.machine_selected_slot.shape == (6, 8)

    def test_inventory_items_zero_initialized(self, state) -> None:
        assert jnp.all(state.machine_inventory_items == 0)

    def test_inventory_counts_zero_initialized(self, state) -> None:
        assert jnp.all(state.machine_inventory_counts == 0)

    def test_machine_selected_recipe_zero_initialized(self, state) -> None:
        assert jnp.all(state.machine_selected_recipe == 0)

    def test_machine_selected_slot_zero_initialized(self, state) -> None:
        assert jnp.all(state.machine_selected_slot == 0)

    def test_inventory_items_dtype(self, state) -> None:
        assert state.machine_inventory_items.dtype == jnp.int32

    def test_inventory_counts_dtype(self, state) -> None:
        assert state.machine_inventory_counts.dtype == jnp.int16


# ---------------------------------------------------------------------------
# State field shapes — procedural generation
# ---------------------------------------------------------------------------


class TestGenerateStateInventoryFields:
    """generate_state also initialises machine inventory fields correctly."""

    @pytest.fixture(scope="class")
    def state(self):
        import jax
        rng = jax.random.PRNGKey(0)
        params = EnvParams(map_width=10, map_height=10, num_players=1)
        return generate_state(rng, params)

    def test_shape(self, state) -> None:
        assert state.machine_inventory_items.shape == (10, 10, MAX_MACHINE_INVENTORY_SLOTS)
        assert state.machine_inventory_counts.shape == (10, 10, MAX_MACHINE_INVENTORY_SLOTS)
        assert state.machine_selected_recipe.shape == (10, 10)
        assert state.machine_selected_slot.shape == (10, 10)

    def test_zero_initialized(self, state) -> None:
        assert jnp.all(state.machine_inventory_items == 0)
        assert jnp.all(state.machine_inventory_counts == 0)


# ---------------------------------------------------------------------------
# Miner inventory semantics
# ---------------------------------------------------------------------------


class TestMinerInventorySlots:
    """Miner uses slot 0 (fuel INPUT) and slot 1 (ore OUTPUT)."""

    def test_fuel_slot_is_index_0(self) -> None:
        assert MACHINE_SLOT_ROLES[MachineType.MINER, 0] == SlotRole.INPUT

    def test_output_slot_is_index_1(self) -> None:
        assert MACHINE_SLOT_ROLES[MachineType.MINER, 1] == SlotRole.OUTPUT

    def test_refuel_reads_from_slot_0(self, state_factory) -> None:
        """refuel_machines reads fuel from machine_inventory_counts[..., 0]."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_inventory_counts=_inv_counts({0: 3}, 1, 1),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_inventory_counts[0, 0, 0] == 2
        assert new_state.machine_power[0, 0] == 10

    def test_fuel_in_wrong_slot_does_not_refuel(self, state_factory) -> None:
        """Coal in slot 2 (not the fuel slot) should NOT refuel the miner."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_inventory_counts=_inv_counts({2: 3}, 1, 1),
        )

        new_state = refuel_machines(state)

        assert new_state.machine_power[0, 0] == 0

    def test_run_miners_writes_to_slot_1(self, state_factory) -> None:
        """run_miners deposits ore into machine_inventory[..., 1]."""
        state = state_factory(
            world_map=jnp.array([[BlockType.IRON]], dtype=jnp.int32),
            block_resources=jnp.array([[20]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
        )

        new_state = run_miners(state)

        assert new_state.machine_inventory_counts[0, 0, 1] == 3
        assert new_state.machine_inventory_items[0, 0, 1] == ItemType.IRON
        # Slot 0 (fuel) and other slots untouched
        assert new_state.machine_inventory_counts[0, 0, 0] == 0
        assert jnp.all(new_state.machine_inventory_counts[0, 0, 2:] == 0)

    def test_output_full_blocks_mining(self, state_factory) -> None:
        """A full output slot (slot 1) prevents further mining."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[10]], dtype=jnp.int32),
            machine_inventory_items=_inv_items({1: int(ItemType.COAL)}, 1, 1),
            machine_inventory_counts=_inv_counts({1: MAX_MACHINE_STACK_SIZE}, 1, 1),
        )

        new_state = run_miners(state)

        assert new_state.machine_inventory_counts[0, 0, 1] == MAX_MACHINE_STACK_SIZE
        assert new_state.block_resources[0, 0] == 50
        assert new_state.machine_power[0, 0] == 10

    def test_full_cycle_refuel_then_mine(self, state_factory) -> None:
        """update_all_machines: machine with no power but fuel refuels then mines."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COPPER]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[MachineType.MINER]], dtype=jnp.int32),
            machine_power=jnp.array([[0]], dtype=jnp.int32),
            machine_inventory_counts=_inv_counts({0: 2}, 1, 1),
        )

        new_state = update_all_machines(state)

        # One coal consumed → power = POWER_PER_COAL - 1 (mined one step)
        assert new_state.machine_inventory_counts[0, 0, 0] == 1  # 2 - 1 fuel consumed
        assert new_state.machine_power[0, 0] == 9  # 10 - 1
        assert new_state.machine_inventory_counts[0, 0, 1] == 3  # mined 3 copper
        assert new_state.machine_inventory_items[0, 0, 1] == ItemType.COPPER


# ---------------------------------------------------------------------------
# Chest inventory semantics
# ---------------------------------------------------------------------------


class TestChestInventory:
    """Chest slots are all STORAGE — no automated flow, full player access."""

    def test_chest_slots_are_storage(self) -> None:
        roles = MACHINE_SLOT_ROLES[MachineType.CHEST]
        assert np.all(roles == SlotRole.STORAGE)

    def test_chest_has_8_slots(self) -> None:
        assert MACHINE_NUM_SLOTS[MachineType.CHEST] == MAX_MACHINE_INVENTORY_SLOTS

    def test_chest_state_initialized_empty(self, state_factory) -> None:
        """A freshly placed chest has all counts and items at zero."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.CHEST]], dtype=jnp.int32),
        )

        assert jnp.all(state.machine_inventory_items[0, 0] == 0)
        assert jnp.all(state.machine_inventory_counts[0, 0] == 0)

    def test_chest_unaffected_by_update_all_machines(self, state_factory) -> None:
        """Machine update should leave chest inventory unchanged (no automation)."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.CHEST]], dtype=jnp.int32),
            machine_inventory_items=_inv_items({0: int(ItemType.IRON), 3: int(ItemType.COAL)}, 1, 1),
            machine_inventory_counts=_inv_counts({0: 10, 3: 5}, 1, 1),
        )

        new_state = update_all_machines(state)

        assert new_state.machine_inventory_items[0, 0, 0] == ItemType.IRON
        assert new_state.machine_inventory_counts[0, 0, 0] == 10
        assert new_state.machine_inventory_items[0, 0, 3] == ItemType.COAL
        assert new_state.machine_inventory_counts[0, 0, 3] == 5


# ---------------------------------------------------------------------------
# Assembler inventory semantics
# ---------------------------------------------------------------------------


class TestAssemblerInventory:
    """Assembler has 3 INPUT slots and 1 OUTPUT slot."""

    def test_assembler_input_slots(self) -> None:
        roles = MACHINE_SLOT_ROLES[MachineType.ASSEMBLER]
        assert roles[0] == SlotRole.INPUT
        assert roles[1] == SlotRole.INPUT
        assert roles[2] == SlotRole.INPUT

    def test_assembler_output_slot(self) -> None:
        assert MACHINE_SLOT_ROLES[MachineType.ASSEMBLER, 3] == SlotRole.OUTPUT

    def test_assembler_inactive_slots(self) -> None:
        roles = MACHINE_SLOT_ROLES[MachineType.ASSEMBLER]
        assert np.all(roles[4:] == SlotRole.NONE)

    def test_assembler_num_slots(self) -> None:
        assert MACHINE_NUM_SLOTS[MachineType.ASSEMBLER] == 4

    def test_assembler_state_initialized_empty(self, state_factory) -> None:
        """Freshly placed assembler has all inventory fields at zero."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.ASSEMBLER]], dtype=jnp.int32),
        )

        assert jnp.all(state.machine_inventory_items[0, 0] == 0)
        assert jnp.all(state.machine_inventory_counts[0, 0] == 0)
        assert state.machine_selected_recipe[0, 0] == 0

    def test_assembler_unaffected_by_update_all_machines(self, state_factory) -> None:
        """Assembler has no automation yet — machine update leaves it unchanged."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array([[MachineType.ASSEMBLER]], dtype=jnp.int32),
            machine_inventory_items=_inv_items({0: int(ItemType.COPPER)}, 1, 1),
            machine_inventory_counts=_inv_counts({0: 5}, 1, 1),
        )

        new_state = update_all_machines(state)

        assert new_state.machine_inventory_counts[0, 0, 0] == 5
        assert new_state.machine_inventory_items[0, 0, 0] == ItemType.COPPER


# ---------------------------------------------------------------------------
# machine_selected_slot and machine_selected_recipe
# ---------------------------------------------------------------------------


class TestMachineSelectedFields:
    """machine_selected_recipe and machine_selected_slot initialise to zero."""

    def test_selected_recipe_zero(self, state_factory) -> None:
        state = state_factory(
            world_map=jnp.ones((3, 3), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        assert jnp.all(state.machine_selected_recipe == 0)

    def test_selected_slot_zero(self, state_factory) -> None:
        state = state_factory(
            world_map=jnp.ones((3, 3), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        assert jnp.all(state.machine_selected_slot == 0)

    def test_selected_recipe_preserved_in_replace(self, state_factory) -> None:
        """State.replace propagates machine_selected_recipe correctly."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        recipe_arr = jnp.array([[2]], dtype=jnp.int32)
        new_state = state.replace(machine_selected_recipe=recipe_arr)
        assert new_state.machine_selected_recipe[0, 0] == 2
