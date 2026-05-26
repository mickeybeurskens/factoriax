"""Tests for the achievement system (entity-based state model)."""

import jax.numpy as jnp

from factoriax import BlockType, ItemType
from factoriax.engine.achievements import (
    ACHIEVEMENT_INFO,
    core_game_conditions,
    count_machines,
    count_total_items,
)
from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    Machine,
)


def _achievement_index(achievement_id: str) -> int:
    """Look up an achievement's index by its string id."""
    for i, info in enumerate(ACHIEVEMENT_INFO):
        if info.id == achievement_id:
            return i
    raise ValueError(f"Unknown achievement: {achievement_id}")


def _player_inv(**items: int) -> jnp.ndarray:
    """Build a single-player pouch inventory."""
    inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        inv = inv.at[0, name_to_type[name]].set(count)
    return inv


class TestItemCounting:
    """Tests for item counting helpers."""

    def test_count_total_items_empty(self, state_factory) -> None:
        """Empty inventory should return zero."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert count_total_items(state, ItemType.COAL) == 0

    def test_count_total_items_single_player(
        self,
        state_factory,
    ) -> None:
        """Should count items in single player pouch."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_player_inv(COAL=10),
        )
        assert count_total_items(state, ItemType.COAL) == 10

    def test_count_total_items_across_players(
        self,
        state_factory,
    ) -> None:
        """Should sum items across multiple players."""
        inv = jnp.zeros((2, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON_ORE].set(5)
        inv = inv.at[1, ItemType.IRON_ORE].set(7)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            num_players=2,
            player_positions=jnp.array(
                [[0, 0], [0, 0]],
                dtype=jnp.int32,
            ),
            player_inventory=inv,
        )
        assert count_total_items(state, ItemType.IRON_ORE) == 12


class TestMachineCounting:
    """Tests for machine counting."""

    def test_count_machines_empty_map(self, state_factory) -> None:
        """No machines should return zero."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert count_machines(state, Machine.MINER) == 0

    def test_count_machines_single(self, state_factory) -> None:
        """Should count single placed miner."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.MINER]],
                dtype=jnp.int32,
            ),
        )
        assert count_machines(state, Machine.MINER) == 1


class TestConditionComputation:
    """Tests for achievement condition computation."""

    def test_no_conditions_met_initially(self, state_factory) -> None:
        """Fresh state should have no conditions met."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        conditions = core_game_conditions(state)
        assert conditions.shape == (MAX_ACHIEVEMENTS,)
        assert not jnp.any(conditions)

    def test_first_ore_condition(self, state_factory) -> None:
        """Mining one ore satisfies First Ore."""
        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.IRON_ORE].set(1)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("first_ore")]

    def test_coal_gathered_condition(self, state_factory) -> None:
        """Holding any coal in player inventory satisfies Coal Gathered."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.COAL].set(1)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=inv,
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("coal_gathered")]

    def test_automated_mining_condition(self, state_factory) -> None:
        """A miner with ore output satisfies Automated Mining."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.MINER]],
                dtype=jnp.int32,
            ),
            buffer_type=jnp.array(
                [[ItemType.IRON_ORE]],
                dtype=jnp.int8,
            ),
            buffer_count=jnp.array([[3]], dtype=jnp.int16),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("automated_mining")]

    def test_first_pipeline_condition(self, state_factory) -> None:
        """A pallet containing items satisfies First Pipeline."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET]],
                dtype=jnp.int32,
            ),
            buffer_type=jnp.array(
                [[ItemType.IRON_ORE]],
                dtype=jnp.int8,
            ),
            buffer_count=jnp.array([[2]], dtype=jnp.int16),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("first_pipeline")]

    def test_assembler_crafted_condition(self, state_factory) -> None:
        """Holding an assembler satisfies Assembler Crafted."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_player_inv(ASSEMBLER=1),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("assembler_crafted")]

    def test_first_assembly_condition(self, state_factory) -> None:
        """Assembler with output satisfies First Assembly."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.ASSEMBLER]],
                dtype=jnp.int32,
            ),
            asm_out_type=jnp.array(
                [[ItemType.IRON_PLATE]],
                dtype=jnp.int8,
            ),
            asm_out_count=jnp.array([[1]], dtype=jnp.int16),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("first_assembly")]

    def test_hull_production_placeholder(self, state_factory) -> None:
        """Hull Production is a placeholder (item removed), always False."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        conditions = core_game_conditions(state)
        assert not conditions[_achievement_index("hull_production")]

    def test_fuel_production_placeholder(self, state_factory) -> None:
        """Fuel Production is a placeholder (item removed), always False."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        conditions = core_game_conditions(state)
        assert not conditions[_achievement_index("fuel_production")]

    def test_rocket_complete_condition(self, state_factory) -> None:
        """Placing a rocket satisfies Rocket Complete."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.ROCKET]],
                dtype=jnp.int32,
            ),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("rocket_complete")]


class TestAchievementUnlocking:
    """Tests for achievement condition latching logic."""

    def test_no_achievements_for_empty_state(
        self,
        state_factory,
    ) -> None:
        """No achievements for a fresh state."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        unlocked = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
        conditions = core_game_conditions(state)
        unlocked = unlocked | conditions
        assert not jnp.any(unlocked)

    def test_achievement_persists(self, state_factory) -> None:
        """Once unlocked, achievements stay unlocked."""
        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.IRON_ORE].set(1)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        unlocked = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
        conditions = core_game_conditions(state)
        unlocked = unlocked | conditions
        idx = _achievement_index("first_ore")
        assert unlocked[idx]
        # Apply again — should still be unlocked.
        unlocked = unlocked | core_game_conditions(state)
        assert unlocked[idx]
