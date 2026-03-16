"""Tests for the achievement system."""

import jax
import jax.numpy as jnp

from factoriax import BlockType, ItemType
from factoriax.achievements import (
    NUM_ACHIEVEMENTS,
    check_achievements,
    compute_all_conditions,
    count_machines,
    count_total_items,
)
from factoriax.constants import NUM_INVENTORY_SLOTS, MachineType


class TestItemCounting:
    """Tests for item counting helpers."""

    def test_count_total_items_empty_inventory(self, state_factory) -> None:
        """Empty inventory should have zero count for all items."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert count_total_items(state, ItemType.COAL) == 0
        assert count_total_items(state, ItemType.IRON) == 0

    def test_count_total_items_single_player(self, state_factory) -> None:
        """Should count items in single player inventory."""
        inv_items = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.COAL)
        inv_counts = jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(10)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        assert count_total_items(state, ItemType.COAL) == 10

    def test_count_total_items_across_players(self, state_factory) -> None:
        """Should sum items across multiple players."""
        inv_items = jnp.zeros((2, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(ItemType.IRON)
        inv_items = inv_items.at[1, 0].set(ItemType.IRON)
        inv_counts = jnp.zeros((2, NUM_INVENTORY_SLOTS), dtype=jnp.int32)
        inv_counts = inv_counts.at[0, 0].set(5)
        inv_counts = inv_counts.at[1, 0].set(7)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            num_players=2,
            player_positions=jnp.array([[0, 0], [0, 0]], dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        assert count_total_items(state, ItemType.IRON) == 12


class TestMachineCounting:
    """Tests for machine counting."""

    def test_count_machines_empty_map(self, state_factory) -> None:
        """Map with no machines should return zero count."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert count_machines(state, MachineType.MINER) == 0

    def test_count_machines_single_miner(self, state_factory) -> None:
        """Should count single placed miner."""
        machine_types = jnp.array([[MachineType.MINER]], dtype=jnp.int32)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=machine_types,
        )
        assert count_machines(state, MachineType.MINER) == 1

    def test_count_machines_multiple_miners(self, state_factory) -> None:
        """Should count multiple placed miners."""
        machine_types = jnp.array(
            [
                [MachineType.MINER, MachineType.NONE],
                [MachineType.NONE, MachineType.MINER],
            ],
            dtype=jnp.int32,
        )
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=machine_types,
        )
        assert count_machines(state, MachineType.MINER) == 2


class TestConditionComputation:
    """Tests for achievement condition computation."""

    def test_no_conditions_met_initially(self, state_factory) -> None:
        """No conditions should be met for fresh state."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        conditions = compute_all_conditions(state)
        assert conditions.shape == (NUM_ACHIEVEMENTS,)
        assert not jnp.any(conditions)

    def test_coal_condition_met(self, state_factory) -> None:
        """Coal collection condition should be met with 1 coal mined."""
        from factoriax.constants import NUM_ITEM_TYPES

        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.COAL].set(1)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        conditions = compute_all_conditions(state)
        assert conditions[0]  # collect_coal_1 is first achievement


class TestAchievementUnlocking:
    """Tests for achievement state updates via check_achievements."""

    def test_no_achievements_unlocked_for_empty_state(self, state_factory) -> None:
        """No achievements should be unlocked for a fresh state."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        new_state = check_achievements(state)
        assert not jnp.any(new_state.achievements_unlocked)

    def test_coal_achievement_unlocked(self, state_factory) -> None:
        """Coal achievement should be unlocked when 1 coal has been mined."""
        from factoriax.constants import NUM_ITEM_TYPES

        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.COAL].set(1)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        new_state = check_achievements(state)
        assert new_state.achievements_unlocked[0]

    def test_already_unlocked_achievement_stays_unlocked(self, state_factory) -> None:
        """Previously unlocked achievements must remain unlocked."""
        already_unlocked = jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_)
        already_unlocked = already_unlocked.at[0].set(True)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            achievements_unlocked=already_unlocked,
        )
        new_state = check_achievements(state)
        assert new_state.achievements_unlocked[0]

    def test_multiple_achievements_unlocked_simultaneously(self, state_factory) -> None:
        """All relevant achievements should unlock in a single call."""
        from factoriax.achievements import ACHIEVEMENT_INFO
        from factoriax.constants import NUM_ITEM_TYPES

        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.COAL].set(1)
        items_mined = items_mined.at[ItemType.IRON].set(1)
        items_mined = items_mined.at[ItemType.COPPER].set(1)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        new_state = check_achievements(state)
        coal_1_idx = next(
            i for i, a in enumerate(ACHIEVEMENT_INFO) if a.id == "collect_coal_1"
        )
        iron_1_idx = next(
            i for i, a in enumerate(ACHIEVEMENT_INFO) if a.id == "collect_iron_1"
        )
        copper_1_idx = next(
            i for i, a in enumerate(ACHIEVEMENT_INFO) if a.id == "collect_copper_1"
        )
        assert new_state.achievements_unlocked[coal_1_idx]
        assert new_state.achievements_unlocked[iron_1_idx]
        assert new_state.achievements_unlocked[copper_1_idx]


class TestJITCompatibility:
    """Tests for JIT compilation compatibility."""

    def test_check_achievements_jit_compatible(self, state_factory) -> None:
        """check_achievements should be JIT-compilable."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        jit_check = jax.jit(check_achievements)
        new_state = jit_check(state)
        assert not jnp.any(new_state.achievements_unlocked)

    def test_check_achievements_jit_unlocks_correctly(self, state_factory) -> None:
        """JIT-compiled check_achievements should unlock achievements correctly."""
        from factoriax.constants import NUM_ITEM_TYPES

        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.COAL].set(1)

        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        jit_check = jax.jit(check_achievements)
        new_state = jit_check(state)
        assert new_state.achievements_unlocked[0]
