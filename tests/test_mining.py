"""Tests for the block resource and mining system (pouch model).

Mining targets the tile in front of the player. To exercise the engine
in these tests the player is placed adjacent to the ore and faced
toward it (e.g. player at (0, 1) with Direction.RIGHT to mine an ore at
(1, 1)).
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from factoriax.engine.constants import BlockType, ItemType
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.constants import BLOCK_MAX_RESOURCES, Direction
from factoriax.engine.game_logic import mine_block
from factoriax.engine.levels import generate_state
from factoriax.engine.tables import MINEABLE_BLOCKS

_PARAMS = EnvParams()


def _ore_state(
    state_factory,
    *,
    ore_resources: int = 5,
    block_type: int = BlockType.COAL,
    facing: int = Direction.RIGHT,
) -> EnvState:
    """Build a 3x3 state with ore at (1, 1) and player adjacent + facing it.

    Default places the player at (0, 1) looking RIGHT toward the ore.
    """
    dx_dy = {
        int(Direction.LEFT): (1, 0),
        int(Direction.RIGHT): (-1, 0),
        int(Direction.UP): (0, 1),
        int(Direction.DOWN): (0, -1),
    }[int(facing)]
    player_xy = (1 + dx_dy[0], 1 + dx_dy[1])
    return state_factory(
        world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
        .at[1, 1]
        .set(block_type),
        player_position=player_xy,
        player_direction=int(facing),
        block_resources=jnp.zeros((3, 3), dtype=jnp.int16).at[1, 1].set(ore_resources),
    )


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
        params = EnvParams()
        state = generate_state(rng, params)
        assert state.block_resources.shape == state.map.shape


class TestMiningResources:
    """Tests for mining with the resource system."""

    def test_mining_decrements_resources(self, state_factory) -> None:
        """Facing ore should decrement resources by 1 (default yield)."""
        state = _ore_state(state_factory, ore_resources=5)
        new = mine_block(state, 0, _PARAMS)
        assert int(new.block_resources[1, 1]) == 4

    def test_mining_yields_item(self, state_factory) -> None:
        """Mining should add one item to the player's pouch."""
        state = _ore_state(state_factory, ore_resources=5)
        new = mine_block(state, 0, _PARAMS)
        assert int(new.player_inventory[0, ItemType.COAL]) == 1

    def test_items_mined_counter_increments(self, state_factory) -> None:
        """Successful mine should bump the global items_mined counter."""
        state = _ore_state(state_factory, ore_resources=5)
        new = mine_block(state, 0, _PARAMS)
        assert int(new.items_mined[ItemType.COAL]) == 1

    def test_block_stays_while_resources_remain(self, state_factory) -> None:
        """Block should remain while resources > 0."""
        state = _ore_state(state_factory, ore_resources=5)
        new = mine_block(state, 0, _PARAMS)
        assert int(new.map[1, 1]) == int(BlockType.COAL)

    def test_block_becomes_dirt_when_depleted(self, state_factory) -> None:
        """Block should become dirt when resources reach zero."""
        state = _ore_state(state_factory, ore_resources=1)
        new = mine_block(state, 0, _PARAMS)
        assert int(new.map[1, 1]) == int(BlockType.DIRT)
        assert int(new.block_resources[1, 1]) == 0
        assert int(new.player_inventory[0, ItemType.COAL]) == 1

    def test_cannot_mine_depleted_block(self, state_factory) -> None:
        """Mining a tile with zero resources should be NOOP."""
        state = _ore_state(state_factory, ore_resources=0)
        new = mine_block(state, 0, _PARAMS)
        assert int(new.player_inventory[0, ItemType.COAL]) == 0
        assert int(new.block_resources[1, 1]) == 0

    @pytest.mark.parametrize(
        "block_type, item_type",
        [
            (BlockType.IRON, ItemType.IRON_ORE),
            (BlockType.COPPER, ItemType.COPPER_ORE),
        ],
        ids=["iron", "copper"],
    )
    def test_mining_ore_type(self, state_factory, block_type, item_type) -> None:
        """Mining should yield the correct ore type."""
        state = _ore_state(state_factory, ore_resources=5, block_type=block_type)
        new = mine_block(state, 0, _PARAMS)
        assert int(new.block_resources[1, 1]) == 4
        assert int(new.player_inventory[0, item_type]) == 1
        assert int(new.map[1, 1]) == int(block_type)


class TestMiningEdgeCases:
    """Edge case tests for mining."""

    def test_mining_non_mineable_in_front_does_nothing(self, state_factory) -> None:
        """Facing dirt should be a NOOP."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(0, 1),
            player_direction=int(Direction.RIGHT),
        )
        new = mine_block(state, 0, _PARAMS)
        assert jnp.all(new.player_inventory[0] == 0)
        assert int(new.map[1, 1]) == int(BlockType.DIRT)

    def test_standing_on_ore_facing_dirt_is_noop(self, state_factory) -> None:
        """Standing on ore but facing a non-ore tile must not mine."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)
            .at[1, 1]
            .set(BlockType.COAL),
            player_position=(1, 1),
            player_direction=int(Direction.LEFT),
            block_resources=jnp.zeros((3, 3), dtype=jnp.int16).at[1, 1].set(5),
        )
        new = mine_block(state, 0, _PARAMS)
        assert int(new.player_inventory[0, ItemType.COAL]) == 0
        assert int(new.block_resources[1, 1]) == 5
        assert int(new.items_mined[ItemType.COAL]) == 0

    @pytest.mark.parametrize(
        "player_xy, facing",
        [
            ((0, 0), Direction.LEFT),
            ((2, 0), Direction.RIGHT),
            ((0, 0), Direction.UP),
            ((0, 2), Direction.DOWN),
        ],
        ids=["left-edge", "right-edge", "top-edge", "bottom-edge"],
    )
    def test_facing_out_of_bounds_is_pytree_equal(
        self, state_factory, player_xy, facing
    ) -> None:
        """Facing OOB on any edge must leave the state pytree-equal."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=player_xy,
            player_direction=int(facing),
        )
        new = mine_block(state, 0, _PARAMS)
        equal_per_leaf = jax.tree.map(
            lambda a, b: bool(jnp.array_equal(a, b)), state, new
        )
        assert all(jax.tree.leaves(equal_per_leaf))

    def test_max_resources_constant_is_30000(self) -> None:
        """BLOCK_MAX_RESOURCES should accommodate the rocket-scenario
        coal patch, which carries 28000 per tile (10x the other ore
        patches at 2800/tile) so four parallel smelter cells have
        enough fuel for the full 8000-tick rocket chain.
        """
        assert BLOCK_MAX_RESOURCES == 30000


class TestPlayerMiningYield:
    """Tests for the per-player mining yield parameter."""

    def test_yield_three_extracts_three(self, state_factory) -> None:
        """With player_mining_yield=3, a successful mine extracts 3."""
        state = _ore_state(state_factory, ore_resources=10)
        params = EnvParams(player_mining_yield=3)
        new = mine_block(state, 0, params)
        assert int(new.player_inventory[0, ItemType.COAL]) == 3
        assert int(new.items_mined[ItemType.COAL]) == 3
        assert int(new.block_resources[1, 1]) == 7

    def test_yield_capped_by_available_resources(self, state_factory) -> None:
        """Yield should cap at remaining tile resources."""
        state = _ore_state(state_factory, ore_resources=2)
        params = EnvParams(player_mining_yield=5)
        new = mine_block(state, 0, params)
        assert int(new.player_inventory[0, ItemType.COAL]) == 2
        assert int(new.block_resources[1, 1]) == 0
        assert int(new.map[1, 1]) == int(BlockType.DIRT)


# NOTE: A JIT cache-size assertion for MINE lives in
# tests/test_jit_retrace.py — it warms up via env.reset_env so the state
# has fully-canonicalized JAX-array leaves. state_factory builds states
# with Python-int leaves (selected_player, timestep) that get promoted
# on first jit, causing a spurious second trace.
