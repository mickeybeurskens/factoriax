"""Tests for the reward functions in factoriax.engine.rewards."""

import jax
import jax.numpy as jnp
import pytest

from factoriax.engine.achievements import achievement_weights
from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.rewards import (
    achievement_reward,
    miner_output_reward,
    mining_reward,
    pallet_filling_reward,
    player_inventory_reward,
    sparse_miner_crafting_reward,
    sparse_mining_reward,
    sparse_pallet_crafting_reward,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.playground.play.achievements import (
    FREE_PLAY_ACHIEVEMENTS,
    free_play_conditions,
)


def _wrap(state: EnvState) -> EnvState:
    """Return *state* with an empty ``achievements_unlocked`` mask."""
    return state.replace(
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )


def _wrap_with(state: EnvState, unlocked: jnp.ndarray) -> EnvState:
    """Return *state* with the given ``achievements_unlocked`` mask."""
    return state.replace(achievements_unlocked=unlocked)


#: A concrete ladder to exercise the reward against. achievement_reward
#: takes weights explicitly. Bit indices mean different things per
#: scenario, so there is no default to fall back on.
_WEIGHTS = achievement_weights(FREE_PLAY_ACHIEVEMENTS)


def _apply_conds(state: EnvState) -> EnvState:
    """OR free-play conditions into the state's ``achievements_unlocked``."""
    conds = free_play_conditions(state)
    return state.replace(
        achievements_unlocked=state.achievements_unlocked | conds,
    )


@pytest.fixture
def params() -> EnvParams:
    """Return default environment parameters for use in reward calls."""
    return EnvParams()


class TestAchievementReward:
    """Tests for achievement_reward."""

    def test_zero_reward_for_identical_states(
        self,
        state_factory,
        params,
    ) -> None:
        """No reward when no new achievements were unlocked."""
        state = _wrap(
            state_factory(
                world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            )
        )
        reward = achievement_reward(state, state, params, _WEIGHTS)
        assert float(reward) == 0.0

    def test_reward_for_newly_unlocked_achievement(
        self,
        state_factory,
        params,
    ) -> None:
        """Reward of 1.0 for a single newly unlocked achievement."""
        prev_state = _wrap(
            state_factory(
                world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            )
        )
        items_mined = (
            jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(1)
        )
        new_state = _apply_conds(
            _wrap(
                state_factory(
                    world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
                    items_mined=items_mined,
                )
            )
        )
        reward = achievement_reward(prev_state, new_state, params, _WEIGHTS)
        assert float(reward) == 1.0

    def test_no_duplicate_reward_for_already_unlocked(
        self,
        state_factory,
        params,
    ) -> None:
        """No reward when the achievement was already unlocked."""
        already = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_).at[0].set(True)
        prev_state = _wrap_with(
            state_factory(
                world_map=jnp.array(
                    [[BlockType.DIRT]],
                    dtype=jnp.int32,
                ),
            ),
            already,
        )
        new_state = _wrap_with(
            state_factory(
                world_map=jnp.array(
                    [[BlockType.DIRT]],
                    dtype=jnp.int32,
                ),
            ),
            already,
        )
        reward = achievement_reward(prev_state, new_state, params, _WEIGHTS)
        assert float(reward) == 0.0

    def test_multiple_achievements_reward(
        self,
        state_factory,
        params,
    ) -> None:
        """Reward equals the number of newly unlocked achievements."""
        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.IRON_ORE].set(10)

        prev_state = _wrap(
            state_factory(
                world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            )
        )
        new_state = _apply_conds(
            _wrap(
                state_factory(
                    world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
                    items_mined=items_mined,
                )
            )
        )
        reward = achievement_reward(prev_state, new_state, params, _WEIGHTS)
        assert float(reward) == 2.0

    def test_jit_compatible(self, state_factory, params) -> None:
        """achievement_reward is JIT-compilable."""
        state = _wrap(
            state_factory(
                world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            )
        )
        jit_fn = jax.jit(achievement_reward)
        reward = jit_fn(state, state, params, _WEIGHTS)
        assert float(reward) == 0.0

    def test_vmap_compatible(self, state_factory, params) -> None:
        """achievement_reward is vmappable over batched states."""
        items_mined = (
            jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(1)
        )
        prev = _wrap(
            state_factory(
                world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
            )
        )
        new = _apply_conds(
            _wrap(
                state_factory(
                    world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
                    items_mined=items_mined,
                )
            )
        )
        batch_prev = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x]),
            prev,
        )
        batch_new = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x]),
            new,
        )
        rewards = jax.vmap(achievement_reward, in_axes=(0, 0, None, None))(
            batch_prev,
            batch_new,
            params,
            _WEIGHTS,
        )
        assert rewards.shape == (2,)
        assert jnp.all(rewards == 1.0)


class TestMiningReward:
    """Tests for mining_reward."""

    def test_proximity_when_standing_on_ore(
        self,
        state_factory,
        params,
    ) -> None:
        """The proximity component is 0.05 when the player is on an ore tile."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        reward = mining_reward(state, state, params)
        assert float(reward) == pytest.approx(0.05)

    def test_proximity_decreases_with_distance(
        self,
        state_factory,
        params,
    ) -> None:
        """Proximity decreases as the player moves away from the ore."""
        world_map = jnp.array(
            [[BlockType.DIRT, BlockType.DIRT, BlockType.COAL]],
            dtype=jnp.int32,
        )
        state_near = state_factory(
            world_map=world_map,
            player_position=(1, 0),
        )
        state_far = state_factory(
            world_map=world_map,
            player_position=(0, 0),
        )
        near_reward = float(mining_reward(state_near, state_near, params))
        far_reward = float(mining_reward(state_far, state_far, params))
        assert near_reward > far_reward

    def test_mining_bonus_for_ore_extracted(
        self,
        state_factory,
        params,
    ) -> None:
        """Bonus of 20.0 per ore item extracted during the step."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev_state = state_factory(world_map=world_map)
        items_mined = (
            jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(1)
        )
        new_state = state_factory(
            world_map=world_map,
            items_mined=items_mined,
        )
        reward = float(mining_reward(prev_state, new_state, params))
        assert reward == pytest.approx(20.0 + 0.05 / 3.0, abs=0.01)

    def test_no_bonus_when_no_ore_mined(
        self,
        state_factory,
        params,
    ) -> None:
        """The mining bonus is zero when items_mined is unchanged."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        reward = float(mining_reward(state, state, params))
        assert reward == pytest.approx(0.05)

    def test_zero_proximity_no_ore_on_map(
        self,
        state_factory,
        params,
    ) -> None:
        """Proximity component near-zero when no ore exists on the map."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        reward = float(mining_reward(state, state, params))
        assert reward < 0.5

    def test_jit_compatible(self, state_factory, params) -> None:
        """mining_reward is JIT-compilable."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            player_position=(0, 0),
        )
        jit_fn = jax.jit(mining_reward)
        reward = jit_fn(state, state, params)
        assert float(reward) == pytest.approx(0.05)

    def test_vmap_compatible(self, state_factory, params) -> None:
        """mining_reward is vmappable over batched states."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        batch = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x]),
            state,
        )
        rewards = jax.vmap(mining_reward, in_axes=(0, 0, None))(
            batch,
            batch,
            params,
        )
        assert rewards.shape == (2,)
        assert jnp.all(rewards == pytest.approx(0.05))

    def test_multiple_ore_types_mined_bonus(
        self,
        state_factory,
        params,
    ) -> None:
        """Mining bonus accumulates across coal, iron, and copper items."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev_state = state_factory(world_map=world_map)
        items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items_mined = items_mined.at[ItemType.COAL].set(1)
        items_mined = items_mined.at[ItemType.IRON_ORE].set(2)
        new_state = state_factory(
            world_map=world_map,
            items_mined=items_mined,
        )
        reward = float(mining_reward(prev_state, new_state, params))
        assert reward == pytest.approx(60.0 + 0.05 / 3.0, abs=0.01)


class TestSparseMiningReward:
    """Tests for sparse_mining_reward."""

    def test_zero_when_nothing_mined(self, state_factory, params) -> None:
        """No reward on steps where items_mined is unchanged."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
        )
        assert float(sparse_mining_reward(state, state, params)) == 0.0

    def test_one_per_ore_item(self, state_factory, params) -> None:
        """Returns 1.0 for each ore item extracted."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev = state_factory(world_map=world_map)
        items = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(3)
        new = state_factory(world_map=world_map, items_mined=items)
        assert float(sparse_mining_reward(prev, new, params)) == pytest.approx(
            3.0,
        )

    def test_accumulates_across_ore_types(
        self,
        state_factory,
        params,
    ) -> None:
        """Delta summed across coal, iron, and copper."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev = state_factory(world_map=world_map)
        items = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        items = items.at[ItemType.COAL].set(1)
        items = items.at[ItemType.IRON_ORE].set(2)
        items = items.at[ItemType.COPPER_ORE].set(3)
        new = state_factory(world_map=world_map, items_mined=items)
        assert float(sparse_mining_reward(prev, new, params)) == pytest.approx(
            6.0,
        )

    def test_no_proximity_component(self, state_factory, params) -> None:
        """Sparse reward is zero when no ore mined, even on ore tile."""
        world_map = jnp.array([[BlockType.COAL]], dtype=jnp.int32)
        state = state_factory(world_map=world_map, player_position=(0, 0))
        assert float(sparse_mining_reward(state, state, params)) == 0.0

    def test_jit_compatible(self, state_factory, params) -> None:
        """sparse_mining_reward is JIT-compilable."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        )
        jit_fn = jax.jit(sparse_mining_reward)
        assert float(jit_fn(state, state, params)) == 0.0


# ---------------------------------------------------------------------------
# Rewards that read the entity buffers
# ---------------------------------------------------------------------------


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
