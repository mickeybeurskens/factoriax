"""Tests for the achievement system (pouch inventory model)."""

import jax.numpy as jnp

from factoriax import BlockType, EnvState, ItemType
from factoriax.achievements import (
    ACHIEVEMENT_INFO,
    core_game_conditions,
    count_machines,
    count_total_items,
)
from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    MachineType,
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


def _machine_inv(
    h: int, w: int, y: int, x: int, **items: int,
) -> jnp.ndarray:
    """Build a machine inventory with items at one tile."""
    inv = jnp.zeros((h, w, NUM_ITEM_TYPES), dtype=jnp.int16)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        inv = inv.at[y, x, name_to_type[name]].set(count)
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
        self, state_factory,
    ) -> None:
        """Should count items in single player pouch."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_player_inv(COAL=10),
        )
        assert count_total_items(state, ItemType.COAL) == 10

    def test_count_total_items_across_players(
        self, state_factory,
    ) -> None:
        """Should sum items across multiple players."""
        inv = jnp.zeros((2, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON].set(5)
        inv = inv.at[1, ItemType.IRON].set(7)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            num_players=2,
            player_positions=jnp.array(
                [[0, 0], [0, 0]], dtype=jnp.int32,
            ),
            player_inventory=inv,
        )
        assert count_total_items(state, ItemType.IRON) == 12


class TestMachineCounting:
    """Tests for machine counting."""

    def test_count_machines_empty_map(self, state_factory) -> None:
        """No machines should return zero."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        assert count_machines(state, MachineType.MINER) == 0

    def test_count_machines_single(self, state_factory) -> None:
        """Should count single placed miner."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.MINER]], dtype=jnp.int32,
            ),
        )
        assert count_machines(state, MachineType.MINER) == 1


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
        items_mined = items_mined.at[ItemType.IRON].set(1)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("first_ore")]

    def test_fueled_up_condition(self, state_factory) -> None:
        """A miner with coal satisfies Fueled Up."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.MINER]], dtype=jnp.int32,
            ),
            machine_inventory=_machine_inv(1, 1, 0, 0, COAL=5),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("fueled_up")]

    def test_automated_mining_condition(self, state_factory) -> None:
        """A miner with ore output satisfies Automated Mining."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.MINER]], dtype=jnp.int32,
            ),
            machine_inventory=_machine_inv(1, 1, 0, 0, IRON=3),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("automated_mining")]

    def test_first_pipeline_condition(self, state_factory) -> None:
        """A pallet containing items satisfies First Pipeline."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.PALLET]], dtype=jnp.int32,
            ),
            machine_inventory=_machine_inv(1, 1, 0, 0, IRON=2),
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
                [[MachineType.ASSEMBLER]], dtype=jnp.int32,
            ),
            machine_inventory=_machine_inv(1, 1, 0, 0, HULL=1),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("first_assembly")]

    def test_hull_production_condition(self, state_factory) -> None:
        """Holding 10 hulls satisfies Hull Production."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_player_inv(HULL=10),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("hull_production")]

    def test_fuel_production_condition(self, state_factory) -> None:
        """Holding 10 fuel packs satisfies Fuel Production."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            player_inventory=_player_inv(FUEL_PACK=10),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("fuel_production")]

    def test_rocket_complete_condition(self, state_factory) -> None:
        """Placing a rocket satisfies Rocket Complete."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.ROCKET]], dtype=jnp.int32,
            ),
        )
        conditions = core_game_conditions(state)
        assert conditions[_achievement_index("rocket_complete")]


def _apply_achievements(state: EnvState) -> EnvState:
    """Apply core game conditions to state (test helper)."""
    conditions = core_game_conditions(state)
    return state.replace(
        achievements_unlocked=state.achievements_unlocked | conditions,
    )


class TestAchievementUnlocking:
    """Tests for achievement state updates."""

    def test_no_achievements_for_empty_state(
        self, state_factory,
    ) -> None:
        """No achievements for a fresh state."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        state = _apply_achievements(state)
        assert not jnp.any(state.achievements_unlocked)

    def test_achievement_persists(self, state_factory) -> None:
        """Once unlocked, achievements stay unlocked."""
        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.IRON].set(1)
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        state = _apply_achievements(state)
        idx = _achievement_index("first_ore")
        assert state.achievements_unlocked[idx]
        # Apply again — should still be unlocked.
        state = _apply_achievements(state)
        assert state.achievements_unlocked[idx]
