"""Tests for the block resource and mining system (pouch model)."""

import jax.numpy as jnp
import pytest
from jax import random

from factoriax import BlockType, EnvParams, EnvState, ItemType
from factoriax.constants import BLOCK_MAX_RESOURCES, MINEABLE_BLOCKS
from factoriax.game_logic import mine_block
from factoriax.levels import generate_state


class TestBlockResources:
    """Tests for block resource initialization."""

    def test_mineable_blocks_have_base_resources(self) -> None:
        """Mineable blocks should have params.base_resources."""
        rng = random.PRNGKey(42)
        params = EnvParams()
        state = generate_state(rng, params)

        for block_type in MINEABLE_BLOCKS:
            mask = state.map == block_type
            if jnp.any(mask):
                resources = state.block_resources[mask]
                assert jnp.all(resources == params.base_resources)

    def test_non_mineable_blocks_have_zero_resources(self) -> None:
        """Non-mineable blocks should have zero resources."""
        rng = random.PRNGKey(42)
        params = EnvParams()
        state = generate_state(rng, params)

        non_mineable = ~jnp.isin(state.map, MINEABLE_BLOCKS)
        assert jnp.all(state.block_resources[non_mineable] == 0)

    def test_block_resources_shape_matches_map(self) -> None:
        """Block resources should have same shape as map."""
        rng = random.PRNGKey(0)
        params = EnvParams(map_width=16, map_height=24)
        state = generate_state(rng, params)
        assert state.block_resources.shape == state.map.shape


class TestMiningResources:
    """Tests for mining with the resource system."""

    @pytest.fixture
    def coal_state(self, state_factory) -> EnvState:
        """State with player on a coal block with 5 resources."""
        return state_factory(
            world_map=jnp.array(
                [
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.COAL, BlockType.DIRT],
                    [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            block_resources=jnp.array(
                [[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=jnp.int16,
            ),
        )

    def test_mining_decrements_resources(
        self, coal_state: EnvState,
    ) -> None:
        """Mining should decrement resources by 1."""
        new = mine_block(coal_state, 0)
        assert new.block_resources[1, 1] == 4

    def test_mining_yields_item(self, coal_state: EnvState) -> None:
        """Mining should add one item to the player's pouch."""
        new = mine_block(coal_state, 0)
        assert new.player_inventory[0, ItemType.COAL] == 1

    def test_block_stays_while_resources_remain(
        self, coal_state: EnvState,
    ) -> None:
        """Block should remain while resources > 0."""
        new = mine_block(coal_state, 0)
        assert new.map[1, 1] == BlockType.COAL

    def test_block_becomes_dirt_when_depleted(
        self, state_factory,
    ) -> None:
        """Block should become dirt when resources reach zero."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.COAL]], dtype=jnp.int32,
            ),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
        )
        new = mine_block(state, 0)
        assert new.map[0, 0] == BlockType.DIRT
        assert new.block_resources[0, 0] == 0
        assert new.player_inventory[0, ItemType.COAL] == 1

    def test_cannot_mine_depleted_block(self, state_factory) -> None:
        """Should not mine a block with zero resources."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.COAL]], dtype=jnp.int32,
            ),
            block_resources=jnp.array([[0]], dtype=jnp.int16),
        )
        new = mine_block(state, 0)
        assert new.player_inventory[0, ItemType.COAL] == 0

    @pytest.mark.parametrize(
        "block_type, item_type",
        [
            (BlockType.IRON, ItemType.IRON),
            (BlockType.COPPER, ItemType.COPPER),
        ],
        ids=["iron", "copper"],
    )
    def test_mining_ore_type(
        self, state_factory, block_type, item_type,
    ) -> None:
        """Mining should yield the correct ore type."""
        state = state_factory(
            world_map=jnp.array(
                [[block_type]], dtype=jnp.int32,
            ),
            block_resources=jnp.array([[5]], dtype=jnp.int16),
        )
        new = mine_block(state, 0)
        assert new.block_resources[0, 0] == 4
        assert new.player_inventory[0, item_type] == 1
        assert new.map[0, 0] == block_type


class TestMiningEdgeCases:
    """Edge case tests for mining."""

    def test_mining_non_mineable_block_does_nothing(
        self, state_factory,
    ) -> None:
        """Mining dirt should have no effect."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]], dtype=jnp.int32,
            ),
        )
        new = mine_block(state, 0)
        assert jnp.all(new.player_inventory[0] == 0)
        assert new.map[0, 0] == BlockType.DIRT

    def test_max_resources_constant_is_1000(self) -> None:
        """BLOCK_MAX_RESOURCES should be 1000."""
        assert BLOCK_MAX_RESOURCES == 1000
