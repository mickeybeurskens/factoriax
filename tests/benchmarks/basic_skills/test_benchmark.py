"""Tests for the basic_skills benchmark.

Covers infrastructure extensions (LevelBuilder.place_machine, player_inventory),
new reward functions, level construction, and scoring.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax import BlockType, Direction, ItemType
from factoriax.benchmarks.basic_skills import BasicSkillsBenchmark
from factoriax.benchmarks.basic_skills.levels import BASIC_SKILLS_LEVELS
from factoriax.benchmarks.basic_skills.scoring import (
    aggregate_scores,
    score_craft,
    score_craft_miners,
    score_deploy_miner,
    score_fill,
    score_mine,
    score_mining_factory,
)
from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_MACHINE_STACK_SIZE,
    MachineType,
)
from factoriax.levels import LevelBuilder, build_state
from factoriax.rewards import chest_filling_reward, sparse_chest_crafting_reward
from factoriax.state import EnvParams

# -----------------------------------------------------------------------
# LevelBuilder.place_machine
# -----------------------------------------------------------------------


class TestPlaceMachine:
    """Tests for the new LevelBuilder.place_machine method."""

    def test_place_chest(self) -> None:
        """Placing a chest should set machine_types and machine_directions."""
        level = (
            LevelBuilder(5, 5)
            .place_machine(2, 3, MachineType.CHEST, Direction.RIGHT)
            .build("test_place")
        )
        assert level.machine_types is not None
        assert level.machine_directions is not None
        assert int(level.machine_types[3, 2]) == MachineType.CHEST
        assert int(level.machine_directions[3, 2]) == Direction.RIGHT

    def test_place_machine_default_direction(self) -> None:
        """Default direction should be 0."""
        level = (
            LevelBuilder(5, 5)
            .place_machine(1, 1, MachineType.MINER)
            .build("test_default_dir")
        )
        assert int(level.machine_directions[1, 1]) == 0

    def test_place_machine_survives_build_state(self) -> None:
        """Machine should appear in the state after build_state."""
        level = (
            LevelBuilder(5, 5)
            .place_machine(4, 4, MachineType.CHEST)
            .build("test_build")
        )
        params = EnvParams(map_width=5, map_height=5, num_players=1)
        state = build_state(level, params)
        assert int(state.machine_types[4, 4]) == MachineType.CHEST

    def test_place_machine_out_of_bounds_raises(self) -> None:
        """Placing outside the map should raise IndexError."""
        builder = LevelBuilder(3, 3)
        try:
            builder.place_machine(5, 0, MachineType.CHEST)
            assert False, "Expected IndexError"
        except IndexError:
            pass


# -----------------------------------------------------------------------
# Player inventory pre-fill
# -----------------------------------------------------------------------


class TestPlayerInventory:
    """Tests for the Level.player_inventory field."""

    def test_player_starts_with_items(self) -> None:
        """Player inventory should be pre-filled from level definition."""
        level = LevelBuilder(5, 5).build("test_inv")
        level.player_inventory = [(int(ItemType.IRON), 30)]
        params = EnvParams(map_width=5, map_height=5, num_players=1)
        state = build_state(level, params)
        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 30

    def test_multiple_items(self) -> None:
        """Multiple slots should be populated in order."""
        level = LevelBuilder(5, 5).build("test_multi")
        level.player_inventory = [
            (int(ItemType.IRON), 10),
            (int(ItemType.COAL), 5),
        ]
        params = EnvParams(map_width=5, map_height=5, num_players=1)
        state = build_state(level, params)
        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 10
        assert int(state.inventory_items[0, 1]) == ItemType.COAL
        assert int(state.inventory_counts[0, 1]) == 5

    def test_no_inventory_means_empty(self) -> None:
        """Default (None) should give empty inventory."""
        level = LevelBuilder(5, 5).build("test_empty")
        params = EnvParams(map_width=5, map_height=5, num_players=1)
        state = build_state(level, params)
        assert int(jnp.sum(state.inventory_counts)) == 0

    def test_multiplayer_all_get_items(self) -> None:
        """Both players should receive the same starting inventory."""
        level = LevelBuilder(10, 5).build("test_mp")
        level.player_inventory = [(int(ItemType.IRON), 20)]
        params = EnvParams(map_width=10, map_height=5, num_players=2)
        state = build_state(level, params)
        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_items[1, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 20
        assert int(state.inventory_counts[1, 0]) == 20


# -----------------------------------------------------------------------
# Reward functions
# -----------------------------------------------------------------------


class TestSparseCraftingReward:
    """Tests for the sparse_chest_crafting_reward function."""

    def test_crafting_chest_gives_reward(self, state_factory) -> None:
        """Reward should fire when a chest appears and iron was consumed."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        # prev: player has 5 iron, no chests
        prev = state_factory(
            world_map=_map,
            inventory_items=jnp.array([[ItemType.IRON] + [0] * 9], dtype=jnp.int32),
            inventory_counts=jnp.array([[5] + [0] * 9], dtype=jnp.int32),
        )
        # new: iron consumed, chest appeared (instant craft)
        new = state_factory(
            world_map=_map,
            inventory_items=jnp.array([[ItemType.CHEST] + [0] * 9], dtype=jnp.int32),
            inventory_counts=jnp.array([[1] + [0] * 9], dtype=jnp.int32),
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        reward = sparse_chest_crafting_reward(prev, new, params)
        assert float(reward) == 1.0

    def test_mining_ore_gives_no_crafting_reward(self, state_factory) -> None:
        """Ore items should not trigger crafting reward."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        prev = state_factory(world_map=_map)
        new = state_factory(
            world_map=_map,
            inventory_items=jnp.array([[ItemType.IRON] + [0] * 9], dtype=jnp.int32),
            inventory_counts=jnp.array([[5] + [0] * 9], dtype=jnp.int32),
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        reward = sparse_chest_crafting_reward(prev, new, params)
        assert float(reward) == 0.0

    def test_crafting_conveyor_belt_gives_no_reward(self, state_factory) -> None:
        """Non-chest crafted items should not trigger reward."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        prev = state_factory(world_map=_map)
        new = state_factory(
            world_map=_map,
            inventory_items=jnp.array(
                [[ItemType.CONVEYOR_BELT] + [0] * 9],
                dtype=jnp.int32,
            ),
            inventory_counts=jnp.array([[3] + [0] * 9], dtype=jnp.int32),
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        reward = sparse_chest_crafting_reward(prev, new, params)
        assert float(reward) == 0.0


class TestChestFillingReward:
    """Tests for the chest_filling_reward function (per-item deposit)."""

    def test_deposit_gives_per_item_reward(self, state_factory) -> None:
        """Depositing 10 items into a chest should give reward 10.0."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        m_types = jnp.full((3, 3), MachineType.NONE, dtype=jnp.int32)
        m_types = m_types.at[1, 1].set(MachineType.CHEST)

        m_counts_prev = jnp.zeros(
            (3, 3, MAX_MACHINE_INVENTORY_SLOTS),
            dtype=jnp.int16,
        )
        m_counts_new = m_counts_prev.at[1, 1, 0].set(10)

        prev = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts_prev,
        )
        new = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts_new,
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        reward = chest_filling_reward(prev, new, params)
        assert float(reward) == 10.0

    def test_single_item_deposit(self, state_factory) -> None:
        """Depositing 1 item into a chest should give reward 1.0."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        m_types = jnp.full((3, 3), MachineType.NONE, dtype=jnp.int32)
        m_types = m_types.at[1, 1].set(MachineType.CHEST)

        m_counts_prev = jnp.zeros(
            (3, 3, MAX_MACHINE_INVENTORY_SLOTS),
            dtype=jnp.int16,
        )
        m_counts_prev = m_counts_prev.at[1, 1, 0].set(63)
        m_counts_new = m_counts_prev.at[1, 1, 0].set(MAX_MACHINE_STACK_SIZE)

        prev = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts_prev,
        )
        new = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts_new,
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        reward = chest_filling_reward(prev, new, params)
        assert float(reward) == 1.0

    def test_non_chest_machine_no_reward(self, state_factory) -> None:
        """Items added to a non-chest machine should give no reward."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        m_types = jnp.full((3, 3), MachineType.NONE, dtype=jnp.int32)
        m_types = m_types.at[1, 1].set(MachineType.MINER)

        m_counts_new = jnp.zeros(
            (3, 3, MAX_MACHINE_INVENTORY_SLOTS),
            dtype=jnp.int16,
        )
        m_counts_new = m_counts_new.at[1, 1, 0].set(10)

        prev = state_factory(world_map=_map, machine_types=m_types)
        new = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts_new,
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        reward = chest_filling_reward(prev, new, params)
        assert float(reward) == 0.0

    def test_no_change_gives_zero_reward(self, state_factory) -> None:
        """No change in chest contents should give zero reward."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        m_types = jnp.full((3, 3), MachineType.NONE, dtype=jnp.int32)
        m_types = m_types.at[1, 1].set(MachineType.CHEST)

        m_counts = jnp.zeros(
            (3, 3, MAX_MACHINE_INVENTORY_SLOTS),
            dtype=jnp.int16,
        )
        m_counts = m_counts.at[1, 1, 0].set(30)

        prev = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts,
        )
        new = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts,
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        reward = chest_filling_reward(prev, new, params)
        assert float(reward) == 0.0


# -----------------------------------------------------------------------
# Level construction
# -----------------------------------------------------------------------


class TestLevelConstruction:
    """Verify that all three levels build and produce valid states."""

    def test_all_levels_build(self) -> None:
        """Every level should build into a valid EnvState."""
        for bl in BASIC_SKILLS_LEVELS:
            state = build_state(bl.level, bl.env_params)
            assert state.map.shape == (
                bl.env_params.map_height,
                bl.env_params.map_width,
            )

    def test_craft_level_has_starting_iron(self) -> None:
        """The craft_chests level should pre-fill player with 30 iron."""
        bl = BASIC_SKILLS_LEVELS[1]
        state = build_state(bl.level, bl.env_params)
        assert int(state.inventory_items[0, 0]) == ItemType.IRON
        assert int(state.inventory_counts[0, 0]) == 30

    def test_fill_level_has_chest(self) -> None:
        """The fill_chest level should have a pre-placed chest."""
        bl = BASIC_SKILLS_LEVELS[2]
        state = build_state(bl.level, bl.env_params)
        assert int(state.machine_types[3, 5]) == MachineType.CHEST


# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------


class TestScoring:
    """Tests for per-level and aggregate scoring."""

    def test_score_mine(self) -> None:
        """Mining score should sum coal and iron."""
        assert score_mine({"coal": 5, "iron": 3}) == 8.0

    def test_score_craft(self, state_factory) -> None:
        """Crafting score should count chest items in inventory."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        state = state_factory(
            world_map=_map,
            inventory_items=jnp.array(
                [[ItemType.CHEST, ItemType.CHEST] + [0] * 8],
                dtype=jnp.int32,
            ),
            inventory_counts=jnp.array([[3, 2] + [0] * 8], dtype=jnp.int32),
        )
        assert score_craft(state) == 5.0

    def test_score_fill(self, state_factory) -> None:
        """Fill score should count full stacks in chests only."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        m_types = jnp.full((3, 3), MachineType.NONE, dtype=jnp.int32)
        m_types = m_types.at[0, 0].set(MachineType.CHEST)
        m_counts = jnp.zeros((3, 3, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
        m_counts = m_counts.at[0, 0, 0].set(MAX_MACHINE_STACK_SIZE)
        m_counts = m_counts.at[0, 0, 1].set(MAX_MACHINE_STACK_SIZE)
        m_counts = m_counts.at[0, 0, 2].set(30)
        state = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts,
        )
        assert score_fill(state) == 2.0

    def test_score_craft_miners(self, state_factory) -> None:
        """Miner-crafting score should count miner items in inventory."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        state = state_factory(
            world_map=_map,
            inventory_items=jnp.array(
                [[ItemType.MINER, ItemType.MINER] + [0] * 8],
                dtype=jnp.int32,
            ),
            inventory_counts=jnp.array([[2, 1] + [0] * 8], dtype=jnp.int32),
        )
        assert score_craft_miners(state) == 3.0

    def test_score_deploy_miner(self, state_factory) -> None:
        """Deploy-miner score should count items in miner output slots."""
        _map = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        m_types = jnp.full((3, 3), MachineType.NONE, dtype=jnp.int32)
        m_types = m_types.at[0, 0].set(MachineType.MINER)
        m_counts = jnp.zeros((3, 3, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16)
        m_counts = m_counts.at[0, 0, 1].set(42)  # slot 1 = output
        state = state_factory(
            world_map=_map,
            machine_types=m_types,
            machine_inventory_counts=m_counts,
        )
        assert score_deploy_miner(state) == 42.0

    def test_score_mining_factory(self) -> None:
        """Mining-factory score should sum all ore types."""
        assert score_mining_factory({"coal": 10, "iron": 20, "copper": 5}) == 35.0

    def test_aggregate_normalises(self) -> None:
        """Aggregate should normalise to [0, 1] per level."""
        scores = {
            "mine_resources": 18.0,
            "craft_chests": 15.0,
            "fill_chest": 8.0,
        }
        assert abs(aggregate_scores(scores) - 1.0) < 1e-6

    def test_aggregate_partial(self) -> None:
        """Partial scores should produce sub-1.0 aggregate."""
        scores = {
            "mine_resources": 9.0,
            "craft_chests": 0.0,
            "fill_chest": 0.0,
        }
        expected = (0.5 + 0.0 + 0.0) / 3.0
        assert abs(aggregate_scores(scores) - expected) < 1e-6


# -----------------------------------------------------------------------
# Benchmark class
# -----------------------------------------------------------------------


class TestBasicSkillsBenchmark:
    """Tests for the BasicSkillsBenchmark class."""

    def test_protocol_compliance(self) -> None:
        """Benchmark should satisfy the Benchmark protocol."""
        from factoriax.benchmarks.core import Benchmark

        b = BasicSkillsBenchmark()
        assert isinstance(b, Benchmark)

    def test_name(self) -> None:
        """Name should be 'basic_skills'."""
        assert BasicSkillsBenchmark().name == "basic_skills"

    def test_num_players(self) -> None:
        """Should require 1 player."""
        assert BasicSkillsBenchmark().num_players == 1

    def test_level_count(self) -> None:
        """Should have exactly eight levels."""
        assert len(BasicSkillsBenchmark().levels()) == 8

    def test_level_names(self) -> None:
        """Level names should match expected values."""
        names = [lvl.name for lvl in BasicSkillsBenchmark().levels()]
        assert names == [
            "mine_resources",
            "craft_chests",
            "fill_chest",
            "craft_miners",
            "deploy_miner",
            "mining_factory",
            "place_and_fuel",
            "withdraw_ore",
        ]
