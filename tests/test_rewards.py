"""Tests for the reward functions in factoriax.rewards."""

import jax
import jax.numpy as jnp
import pytest

from factoriax import BlockType, ItemType
from factoriax.achievements import NUM_ACHIEVEMENTS
from factoriax.constants import NUM_ITEM_TYPES
from factoriax.rewards import achievement_reward, mining_reward, sparse_mining_reward
from factoriax.state import EnvParams


@pytest.fixture
def params() -> EnvParams:
    """Return default environment parameters for use in reward calls."""
    return EnvParams()


class TestAchievementReward:
    """Tests for achievement_reward."""

    def test_zero_reward_for_identical_states(self, state_factory, params) -> None:
        """No reward when no new achievements were unlocked."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        reward = achievement_reward(state, state, params)
        assert float(reward) == 0.0

    def test_reward_for_newly_unlocked_achievement(self, state_factory, params) -> None:
        """Reward of 1.0 for a single newly unlocked achievement."""
        from factoriax.constants import NUM_ITEM_TYPES

        prev_state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        items_mined = (
            jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(1)
        )
        new_state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            items_mined=items_mined,
        )
        # Unlock the coal achievement on new_state
        from factoriax.achievements import check_achievements

        new_state = check_achievements(new_state)
        reward = achievement_reward(prev_state, new_state, params)
        assert float(reward) == 1.0

    def test_no_duplicate_reward_for_already_unlocked(
        self, state_factory, params
    ) -> None:
        """No reward when the achievement was already unlocked in prev_state."""
        already = jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_).at[0].set(True)
        prev_state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            achievements_unlocked=already,
        )
        new_state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            achievements_unlocked=already,
        )
        reward = achievement_reward(prev_state, new_state, params)
        assert float(reward) == 0.0

    def test_multiple_achievements_reward(self, state_factory, params) -> None:
        """Reward equals the number of newly unlocked achievements."""
        from factoriax.achievements import check_achievements

        # Mining 10 total ores unlocks both First Ore and Stockpile.
        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.IRON].set(10)

        prev_state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        new_state = check_achievements(
            state_factory(
                world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
                items_mined=items_mined,
            )
        )
        reward = achievement_reward(prev_state, new_state, params)
        assert float(reward) == 2.0

    def test_jit_compatible(self, state_factory, params) -> None:
        """achievement_reward should be JIT-compilable."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        jit_fn = jax.jit(achievement_reward)
        reward = jit_fn(state, state, params)
        assert float(reward) == 0.0

    def test_vmap_compatible(self, state_factory, params) -> None:
        """achievement_reward should be vmappable over batched states."""
        from factoriax.achievements import check_achievements

        items_mined = (
            jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(1)
        )
        prev = state_factory(world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32))
        new = check_achievements(
            state_factory(
                world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
                items_mined=items_mined,
            )
        )
        # Batch both states along a leading axis
        batch_prev = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), prev)
        batch_new = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), new)
        rewards = jax.vmap(achievement_reward, in_axes=(0, 0, None))(
            batch_prev, batch_new, params
        )
        assert rewards.shape == (2,)
        assert jnp.all(rewards == 1.0)


class TestMiningReward:
    """Tests for mining_reward."""

    def test_proximity_when_standing_on_ore(self, state_factory, params) -> None:
        """Proximity component should be 1.0 when player is on an ore tile."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        reward = mining_reward(state, state, params)
        # At dist=0: proximity=1/(1+0)=1.0, no mining delta -> reward=1.0
        assert float(reward) == pytest.approx(1.0)

    def test_proximity_decreases_with_distance(self, state_factory, params) -> None:
        """Proximity should decrease as player moves away from ore."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.DIRT, BlockType.COAL],
            ],
            dtype=jnp.int32,
        )
        state_near = state_factory(world_map=world_map, player_position=(1, 0))
        state_far = state_factory(world_map=world_map, player_position=(0, 0))
        near_reward = float(mining_reward(state_near, state_near, params))
        far_reward = float(mining_reward(state_far, state_far, params))
        assert near_reward > far_reward

    def test_mining_bonus_for_ore_extracted(self, state_factory, params) -> None:
        """Bonus of 5.0 per ore item extracted during the step."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev_state = state_factory(world_map=world_map)
        items_mined = (
            jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(1)
        )
        new_state = state_factory(world_map=world_map, items_mined=items_mined)
        # 1x1 DIRT map: sentinel = map_h + map_w = 2, proximity = 1/(1+2) = 1/3
        # mining_bonus = 5.0 * 1 = 5.0; total = 5.0 + 1/3
        reward = float(mining_reward(prev_state, new_state, params))
        assert reward == pytest.approx(5.0 + 1.0 / 3.0, abs=0.01)

    def test_no_bonus_when_no_ore_mined(self, state_factory, params) -> None:
        """Mining bonus should be zero when items_mined is unchanged."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        reward = float(mining_reward(state, state, params))
        # proximity only: 1/(1+0)=1.0
        assert reward == pytest.approx(1.0)

    def test_zero_proximity_no_ore_on_map(self, state_factory, params) -> None:
        """Proximity component near-zero when no ore exists on the map."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        reward = float(mining_reward(state, state, params))
        # proximity = 1 / (1 + 1+1) = 1/3 ~ 0.333 for 1x1 map sentinel
        assert reward < 0.5

    def test_jit_compatible(self, state_factory, params) -> None:
        """mining_reward should be JIT-compilable."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            player_position=(0, 0),
        )
        jit_fn = jax.jit(mining_reward)
        reward = jit_fn(state, state, params)
        assert float(reward) == pytest.approx(1.0)

    def test_vmap_compatible(self, state_factory, params) -> None:
        """mining_reward should be vmappable over batched states."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
        rewards = jax.vmap(mining_reward, in_axes=(0, 0, None))(batch, batch, params)
        assert rewards.shape == (2,)
        assert jnp.all(rewards == pytest.approx(1.0))

    def test_multiple_ore_types_mined_bonus(self, state_factory, params) -> None:
        """Mining bonus accumulates across coal, iron, and copper items."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev_state = state_factory(world_map=world_map)
        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.COAL].set(1)
        items_mined = items_mined.at[ItemType.IRON].set(2)
        new_state = state_factory(world_map=world_map, items_mined=items_mined)
        # 1x1 DIRT map: sentinel = 2, proximity = 1/3
        # mining_bonus = 5 * (1 + 2) = 15; total = 15.0 + 1/3
        reward = float(mining_reward(prev_state, new_state, params))
        assert reward == pytest.approx(15.0 + 1.0 / 3.0, abs=0.01)


class TestSparseMiningReward:
    """Tests for sparse_mining_reward."""

    def test_zero_when_nothing_mined(self, state_factory, params) -> None:
        """No reward on steps where items_mined is unchanged."""
        state = state_factory(world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32))
        assert float(sparse_mining_reward(state, state, params)) == 0.0

    def test_one_per_ore_item(self, state_factory, params) -> None:
        """Returns 1.0 for each ore item extracted."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev = state_factory(world_map=world_map)
        items = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(3)
        new = state_factory(world_map=world_map, items_mined=items)
        assert float(sparse_mining_reward(prev, new, params)) == pytest.approx(3.0)

    def test_accumulates_across_ore_types(self, state_factory, params) -> None:
        """Delta summed across coal, iron, and copper."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev = state_factory(world_map=world_map)
        items = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items = items.at[ItemType.COAL].set(1)
        items = items.at[ItemType.IRON].set(2)
        items = items.at[ItemType.COPPER].set(3)
        new = state_factory(world_map=world_map, items_mined=items)
        assert float(sparse_mining_reward(prev, new, params)) == pytest.approx(6.0)

    def test_no_proximity_component(self, state_factory, params) -> None:
        """Sparse reward is zero when no ore mined, even when player is on ore."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        assert float(sparse_mining_reward(state, state, params)) == 0.0

    def test_jit_compatible(self, state_factory, params) -> None:
        """sparse_mining_reward should be JIT-compilable."""
        state = state_factory(world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32))
        jit_fn = jax.jit(sparse_mining_reward)
        assert float(jit_fn(state, state, params)) == 0.0
