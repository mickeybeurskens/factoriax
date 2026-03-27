"""Tests for game logic orchestration functions.

Covers the private helpers ``_find_deposit_slot`` and ``_find_withdraw_slot``
(slot priority and recipe filtering), the action dispatcher
``_handle_player_action``, and the top-level ``factoriax_step``.
"""

import jax
import jax.numpy as jnp
import pytest

from factoriax import Action, BlockType, EnvParams, ItemType
from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_MACHINE_STACK_SIZE,
    MachineType,
)
from factoriax.game_logic import (
    _find_deposit_slot,
    _find_withdraw_slot,
    _handle_player_action,
    factoriax_step,
)

# ---------------------------------------------------------------------------
# _find_deposit_slot
# ---------------------------------------------------------------------------


class TestFindDepositSlot:
    """Unit tests for deposit slot priority and filtering logic."""

    def _chest_args(
        self,
        slot_items: list[int],
        slot_counts: list[int],
        item: int,
        focused_slot: int = 0,
    ) -> tuple[jax.Array, ...]:
        """Build arguments for _find_deposit_slot on a CHEST machine.

        Chests have 8 STORAGE slots, no recipe filtering.

        Args:
            slot_items: Item type per slot.
            slot_counts: Stack count per slot.
            item: Item the player wants to deposit.
            focused_slot: UI-selected slot index.

        Returns:
            Tuple ready to unpack into _find_deposit_slot.
        """
        n_pad = MAX_MACHINE_INVENTORY_SLOTS - len(slot_items)
        padded_items = slot_items + [0] * n_pad
        padded_counts = slot_counts + [0] * (
            MAX_MACHINE_INVENTORY_SLOTS - len(slot_counts)
        )
        return (
            jnp.int32(MachineType.CHEST),
            jnp.array(padded_items, dtype=jnp.int32),
            jnp.array(padded_counts, dtype=jnp.int16),
            jnp.int32(8),
            jnp.int32(item),
            jnp.int32(0),
            jnp.int32(focused_slot),
        )

    def test_matching_slot_preferred_over_empty(self) -> None:
        """A slot already holding the same item beats an empty slot."""
        best, has_slot, _ = _find_deposit_slot(
            *self._chest_args(
                slot_items=[0, ItemType.IRON, 0],
                slot_counts=[0, 5, 0],
                item=ItemType.IRON,
            )
        )
        assert bool(has_slot)
        assert int(best) == 1

    def test_focused_matching_beats_unfocused_matching(self) -> None:
        """Focused slot with a match beats an unfocused match."""
        best, has_slot, _ = _find_deposit_slot(
            *self._chest_args(
                slot_items=[ItemType.IRON, ItemType.IRON],
                slot_counts=[3, 3],
                item=ItemType.IRON,
                focused_slot=1,
            )
        )
        assert bool(has_slot)
        assert int(best) == 1

    def test_focused_empty_beats_unfocused_empty(self) -> None:
        """Focused empty slot beats a lower-index empty slot."""
        best, has_slot, _ = _find_deposit_slot(
            *self._chest_args(
                slot_items=[0, 0, 0],
                slot_counts=[0, 0, 0],
                item=ItemType.COAL,
                focused_slot=2,
            )
        )
        assert bool(has_slot)
        assert int(best) == 2

    def test_no_usable_slot_returns_false(self) -> None:
        """When all slots are full, has_slot should be False."""
        best, has_slot, _ = _find_deposit_slot(
            *self._chest_args(
                slot_items=[ItemType.COPPER] * 8,
                slot_counts=[MAX_MACHINE_STACK_SIZE] * 8,
                item=ItemType.IRON,
            )
        )
        assert not bool(has_slot)

    def test_assembler_recipe_filters_input_slots(self) -> None:
        """Assembler INPUT slots should only accept items matching the recipe.

        Recipe 0 (Hull) requires iron. Depositing copper into an INPUT slot
        should fail even though the slot is empty.
        """
        slot_items = [0] * MAX_MACHINE_INVENTORY_SLOTS
        slot_counts = [0] * MAX_MACHINE_INVENTORY_SLOTS
        # Assembler has 3 INPUT + 1 OUTPUT + 4 NONE, num_slots=4
        best, has_slot, _ = _find_deposit_slot(
            jnp.int32(MachineType.ASSEMBLER),
            jnp.array(slot_items, dtype=jnp.int32),
            jnp.array(slot_counts, dtype=jnp.int16),
            jnp.int32(4),
            jnp.int32(ItemType.COPPER),  # copper is not in hull recipe
            jnp.int32(0),  # recipe 0 = hull (requires iron only)
            jnp.int32(0),
        )
        # Copper cannot go into INPUT slots for the hull recipe, and OUTPUT
        # is not a deposit role, so no slot is usable.
        assert not bool(has_slot)


# ---------------------------------------------------------------------------
# _find_withdraw_slot
# ---------------------------------------------------------------------------


class TestFindWithdrawSlot:
    """Unit tests for withdraw slot priority logic."""

    def _miner_args(
        self,
        slot_items: list[int],
        slot_counts: list[int],
        focused_slot: int = 0,
    ) -> tuple[jax.Array, ...]:
        """Build arguments for _find_withdraw_slot on a MINER.

        Miners have slot 0 = INPUT (fuel) and slot 1 = OUTPUT (ore).

        Args:
            slot_items: Item type per slot (at least 2 entries).
            slot_counts: Stack count per slot.
            focused_slot: UI-selected slot index.

        Returns:
            Tuple ready to unpack into _find_withdraw_slot.
        """
        n_pad = MAX_MACHINE_INVENTORY_SLOTS - len(slot_items)
        padded_items = slot_items + [0] * n_pad
        padded_counts = slot_counts + [0] * (
            MAX_MACHINE_INVENTORY_SLOTS - len(slot_counts)
        )
        return (
            jnp.int32(MachineType.MINER),
            jnp.array(padded_items, dtype=jnp.int32),
            jnp.array(padded_counts, dtype=jnp.int16),
            jnp.int32(2),
            jnp.int32(focused_slot),
        )

    def test_output_role_preferred_over_input(self) -> None:
        """OUTPUT slot should be chosen over INPUT even at higher index."""
        best, has_source, item, count = _find_withdraw_slot(
            *self._miner_args(
                slot_items=[ItemType.COAL, ItemType.IRON],
                slot_counts=[5, 10],
            )
        )
        assert bool(has_source)
        assert int(best) == 1  # OUTPUT slot
        assert int(item) == ItemType.IRON

    def test_focused_wins_within_same_role(self) -> None:
        """Within the same role, the focused slot should win ties.

        Chests have 8 STORAGE slots so role priority is equal.
        """
        slot_items = [ItemType.IRON, ItemType.COPPER] + [0] * 6
        slot_counts = [10, 10] + [0] * 6
        best, has_source, _, _ = _find_withdraw_slot(
            jnp.int32(MachineType.CHEST),
            jnp.array(slot_items, dtype=jnp.int32),
            jnp.array(slot_counts, dtype=jnp.int16),
            jnp.int32(8),
            jnp.int32(1),  # focus slot 1
        )
        assert bool(has_source)
        assert int(best) == 1

    def test_role_priority_overrides_focus(self) -> None:
        """OUTPUT beats focused INPUT even when INPUT is focused."""
        best, has_source, item, _ = _find_withdraw_slot(
            *self._miner_args(
                slot_items=[ItemType.COAL, ItemType.IRON],
                slot_counts=[5, 3],
                focused_slot=0,  # focus the INPUT slot
            )
        )
        assert bool(has_source)
        assert int(best) == 1  # OUTPUT still wins
        assert int(item) == ItemType.IRON

    def test_no_items_returns_false(self) -> None:
        """When all slots are empty, has_source should be False."""
        _, has_source, _, _ = _find_withdraw_slot(
            *self._miner_args(slot_items=[0, 0], slot_counts=[0, 0])
        )
        assert not bool(has_source)


# ---------------------------------------------------------------------------
# _handle_player_action / factoriax_step
# ---------------------------------------------------------------------------


class TestHandlePlayerAction:
    """Tests for the action dispatch function."""

    @pytest.mark.parametrize(
        "action, expected_pos",
        [
            (Action.LEFT, [0, 1]),
            (Action.RIGHT, [2, 1]),
            (Action.UP, [1, 0]),
            (Action.DOWN, [1, 2]),
        ],
        ids=["left", "right", "up", "down"],
    )
    def test_movement_dispatches_correctly(
        self, state_factory, action: int, expected_pos: list[int]
    ) -> None:
        """Movement actions should update the player's position."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
        )
        new_state = _handle_player_action(state, action, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array(expected_pos))

    def test_mine_decrements_resources(self, state_factory) -> None:
        """MINE action should extract a resource from the block underfoot."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[10]], dtype=jnp.int16),
        )
        new_state = _handle_player_action(state, Action.MINE, 0)
        assert int(new_state.block_resources[0, 0]) == 9
        assert int(new_state.inventory_counts[0, 0]) == 1

    def test_noop_preserves_state(self, state_factory) -> None:
        """NOOP should not change position or inventory."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        new_state = _handle_player_action(state, Action.NOOP, 0)
        assert jnp.array_equal(new_state.player_positions, state.player_positions)
        assert jnp.array_equal(new_state.inventory_counts, state.inventory_counts)


class TestFactoriaxStep:
    """Tests for the top-level environment step function."""

    def test_increments_timestep(self, state_factory) -> None:
        """Each step should increment the timestep by 1."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        rng = jax.random.PRNGKey(0)
        params = EnvParams(map_width=1, map_height=1)
        new_state = factoriax_step(rng, state, Action.NOOP, params)
        assert int(new_state.timestep) == int(state.timestep) + 1

    def test_mine_and_machines_in_single_step(self, state_factory) -> None:
        """A step should run player action, crafting, and machines.

        Set up a state where the player mines coal and a miner also runs.
        Both effects should be visible after one step.
        """
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.COAL, BlockType.IRON]],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[10, 50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.MINER]], dtype=jnp.int32
            ),
            machine_power=jnp.array([[0, 10]], dtype=jnp.int32),
        )
        rng = jax.random.PRNGKey(0)
        params = EnvParams(map_width=2, map_height=1)
        new_state = factoriax_step(rng, state, Action.MINE, params)

        # Player mined coal from (0,0).
        assert int(new_state.block_resources[0, 0]) == 9
        # Miner at (1,0) also ran and extracted iron.
        assert int(new_state.block_resources[0, 1]) < 50
