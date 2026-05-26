"""Regression tests for reward functions that read from entity buffers.

These tests verify that reward functions correctly read from
player_inventory and entity buffer arrays after the entity migration.
"""

import jax.numpy as jnp
import pytest

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    ItemType,
    Machine,
)
from factoriax.engine.rewards import (
    miner_output_reward,
    pallet_filling_reward,
    player_inventory_reward,
    sparse_miner_crafting_reward,
    sparse_pallet_crafting_reward,
)
from factoriax.engine.state import EnvParams


@pytest.fixture
def params() -> EnvParams:
    """Return default environment parameters."""
    return EnvParams()


class TestSparsePalletCraftingReward:
    """Tests for sparse_pallet_crafting_reward reading player_inventory."""

    def test_zero_when_no_change(self, state_factory, params) -> None:
        """Identical states produce zero reward."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        reward = sparse_pallet_crafting_reward(state, state, params)
        assert float(reward) == 0.0

    def test_positive_when_pallet_gained_iron_lost(
        self,
        state_factory,
        params,
    ) -> None:
        """Gaining pallets while losing iron triggers positive reward."""
        inv_before = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv_before = inv_before.at[0, int(ItemType.IRON_ORE)].set(10)
        prev = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv_before,
        )
        inv_after = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv_after = inv_after.at[0, int(ItemType.IRON_ORE)].set(6)
        inv_after = inv_after.at[0, int(ItemType.PALLET)].set(1)
        new = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv_after,
        )
        reward = sparse_pallet_crafting_reward(prev, new, params)
        assert float(reward) == 1.0

    def test_zero_when_only_pallet_gained(
        self,
        state_factory,
        params,
    ) -> None:
        """Gaining pallets without losing iron yields zero (pickup, not craft)."""
        prev = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        inv_after = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv_after = inv_after.at[0, int(ItemType.PALLET)].set(1)
        new = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv_after,
        )
        reward = sparse_pallet_crafting_reward(prev, new, params)
        assert float(reward) == 0.0


class TestSparseMinerCraftingReward:
    """Tests for sparse_miner_crafting_reward reading player_inventory."""

    def test_positive_when_miner_gained_resources_lost(
        self,
        state_factory,
        params,
    ) -> None:
        """Gaining miner while losing iron and copper triggers reward."""
        inv_before = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv_before = inv_before.at[0, int(ItemType.IRON_ORE)].set(10)
        inv_before = inv_before.at[0, int(ItemType.COPPER_ORE)].set(10)
        prev = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv_before,
        )
        inv_after = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv_after = inv_after.at[0, int(ItemType.IRON_ORE)].set(6)
        inv_after = inv_after.at[0, int(ItemType.COPPER_ORE)].set(6)
        inv_after = inv_after.at[0, int(ItemType.MINER)].set(1)
        new = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv_after,
        )
        reward = sparse_miner_crafting_reward(prev, new, params)
        assert float(reward) == 1.0


class TestMinerOutputReward:
    """Tests for miner_output_reward reading entity buffers."""

    def test_zero_when_no_miners(self, state_factory, params) -> None:
        """No miners on the map means zero reward."""
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        reward = miner_output_reward(state, state, params)
        assert float(reward) == 0.0

    def test_positive_when_miner_produces_ore(
        self,
        state_factory,
        params,
    ) -> None:
        """Ore increase in a miner's buffer produces reward."""
        mt = jnp.zeros((4, 4), dtype=jnp.int32)
        mt = mt.at[0, 0].set(int(Machine.MINER))
        prev = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=mt,
        )
        buf_type = jnp.zeros((4, 4), dtype=jnp.int8)
        buf_type = buf_type.at[0, 0].set(int(ItemType.COAL))
        buf_count = jnp.zeros((4, 4), dtype=jnp.int16)
        buf_count = buf_count.at[0, 0].set(3)
        new = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=mt,
            buffer_type=buf_type,
            buffer_count=buf_count,
        )
        reward = miner_output_reward(prev, new, params)
        assert float(reward) == 3.0


class TestPalletFillingReward:
    """Tests for pallet_filling_reward reading entity buffers."""

    def test_positive_when_pallet_gains_items(
        self,
        state_factory,
        params,
    ) -> None:
        """Items deposited into a pallet produce positive reward."""
        mt = jnp.zeros((4, 4), dtype=jnp.int32)
        mt = mt.at[1, 1].set(int(Machine.PALLET))
        prev = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=mt,
        )
        buf_type = jnp.zeros((4, 4), dtype=jnp.int8)
        buf_type = buf_type.at[1, 1].set(int(ItemType.IRON_ORE))
        buf_count = jnp.zeros((4, 4), dtype=jnp.int16)
        buf_count = buf_count.at[1, 1].set(5)
        new = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=mt,
            buffer_type=buf_type,
            buffer_count=buf_count,
        )
        reward = pallet_filling_reward(prev, new, params)
        assert float(reward) == 5.0


class TestPlayerInventoryReward:
    """Tests for player_inventory_reward reading player_inventory."""

    def test_positive_when_items_gained(
        self,
        state_factory,
        params,
    ) -> None:
        """Gaining items produces positive reward."""
        prev = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        inv_after = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv_after = inv_after.at[0, int(ItemType.COAL)].set(7)
        new = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv_after,
        )
        reward = player_inventory_reward(prev, new, params)
        assert float(reward) == 7.0

    def test_zero_when_no_change(self, state_factory, params) -> None:
        """Identical states produce zero reward."""
        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        reward = player_inventory_reward(state, state, params)
        assert float(reward) == 0.0
