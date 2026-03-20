"""Tests for the block resource and mining system."""

import jax.numpy as jnp
import pytest
from jax import random

from factoriax import BlockType, EnvParams, EnvState, ItemType
from factoriax.constants import BLOCK_MAX_RESOURCES, MINEABLE_BLOCKS
from factoriax.game_logic import mine_block
from factoriax.world_gen import generate_world


class TestBlockResources:
    """Tests for block resource initialization."""

    def test_mineable_blocks_have_base_resources(self) -> None:
        """Mineable blocks should be initialized with params.base_resources."""
        rng = random.PRNGKey(42)
        params = EnvParams()
        state = generate_world(rng, params)

        for block_type in MINEABLE_BLOCKS:
            mask = state.map == block_type
            if jnp.any(mask):
                resources_at_block = state.block_resources[mask]
                assert jnp.all(resources_at_block == params.base_resources)

    def test_non_mineable_blocks_have_zero_resources(self) -> None:
        """Non-mineable blocks should have zero resources."""
        rng = random.PRNGKey(42)
        params = EnvParams()
        state = generate_world(rng, params)

        non_mineable_mask = ~jnp.isin(state.map, MINEABLE_BLOCKS)
        resources_at_non_mineable = state.block_resources[non_mineable_mask]
        assert jnp.all(resources_at_non_mineable == 0)

    def test_block_resources_shape_matches_map(self) -> None:
        """Block resources array should have same shape as map."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=16, map_height=24)
        state = generate_world(rng, params)

        assert state.block_resources.shape == state.map.shape


class TestMiningResources:
    """Tests for mining with the resource system."""

    @pytest.fixture
    def coal_state(self, state_factory) -> EnvState:
        """Create a state with player on a coal block with 5 resources."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                [BlockType.DIRT, BlockType.COAL, BlockType.DIRT],
                [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
            ],
            dtype=jnp.int32,
        )
        block_resources = jnp.array(
            [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
            dtype=jnp.int16,
        )
        return state_factory(
            world_map=world_map,
            player_position=(1, 1),
            block_resources=block_resources,
        )

    def test_mining_decrements_resources(self, coal_state: EnvState) -> None:
        """Mining should decrement the block's resources by 1."""
        new_state = mine_block(coal_state, 0)
        assert new_state.block_resources[1, 1] == 4

    def test_mining_yields_item(self, coal_state: EnvState) -> None:
        """Each mining action should yield one item."""
        new_state = mine_block(coal_state, 0)
        assert new_state.inventory_counts[0, 0] == 1
        assert new_state.inventory_items[0, 0] == ItemType.COAL

    def test_block_stays_while_resources_remain(self, coal_state: EnvState) -> None:
        """Block should remain coal while resources are above zero."""
        state = mine_block(coal_state, 0)
        assert state.map[1, 1] == BlockType.COAL
        assert state.block_resources[1, 1] == 4

    def test_block_becomes_dirt_when_depleted(self, state_factory) -> None:
        """Block should become dirt when resources reach zero."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
        )

        new_state = mine_block(state, 0)

        assert new_state.map[0, 0] == BlockType.DIRT
        assert new_state.block_resources[0, 0] == 0
        assert new_state.inventory_counts[0, 0] == 1

    def test_cannot_mine_depleted_block(self, state_factory) -> None:
        """Should not be able to mine a block with zero resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0]], dtype=jnp.int16),
        )

        new_state = mine_block(state, 0)

        assert new_state.inventory_counts[0, 0] == 0
        assert new_state.block_resources[0, 0] == 0

    def test_mining_iron_and_copper(self, state_factory) -> None:
        """Iron and copper blocks should work the same as coal."""
        for block_type, item_type in [
            (BlockType.IRON, ItemType.IRON),
            (BlockType.COPPER, ItemType.COPPER),
        ]:
            state = state_factory(
                world_map=jnp.array([[block_type]], dtype=jnp.int32),
                block_resources=jnp.array([[5]], dtype=jnp.int16),
            )

            new_state = mine_block(state, 0)

            assert new_state.block_resources[0, 0] == 4
            assert new_state.inventory_items[0, 0] == item_type
            assert new_state.inventory_counts[0, 0] == 1
            assert new_state.map[0, 0] == block_type


class TestMiningEdgeCases:
    """Edge case tests for mining."""

    def test_mining_non_mineable_block_does_nothing(self, state_factory) -> None:
        """Mining dirt should have no effect."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )

        new_state = mine_block(state, 0)

        assert new_state.inventory_counts[0, 0] == 0
        assert new_state.map[0, 0] == BlockType.DIRT

    def test_max_resources_constant_is_1000(self) -> None:
        """BLOCK_MAX_RESOURCES should be 1000."""
        assert BLOCK_MAX_RESOURCES == 1000
