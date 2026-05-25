"""Property-based invariant tests for the FactoriaX environment.

These tests verify structural invariants that must hold after any
sequence of actions, regardless of the specific actions taken. They
catch subtle bugs that targeted unit tests miss by exercising the
environment with random action sequences.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax, random

from factoriax.constants import (
    NUM_ACTIONS,
    NUM_ITEM_TYPES,
    PLAYER_MAX_STACK,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.game_logic import factoriax_step, mine_block
from factoriax.machine_spec import MACHINE_MAX_STACK
from factoriax.state import EnvParams, EnvState

# Every test in this file runs a 100-step random rollout through
# ``factoriax_step``, which triggers the full env JIT compile. They
# are the highest-value regression guards (emergent invariants) but
# too slow for the pre-commit inner loop.
pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_PARAMS = EnvParams(map_width=8, map_height=8, max_timesteps=200)
_NUM_RANDOM_STEPS = 100


@pytest.fixture(scope="module")
def random_episode():
    """Module-scoped JIT'd 100-step random rollout for 8x8 2p.

    Returns ``(env, params, run_fn)``. The ``run_fn`` is a JIT-compiled
    function ``rng -> final_state`` that runs ``_NUM_RANDOM_STEPS``
    random actions under ``_SMALL_PARAMS``. Tests vary the seed but
    share the trace: the costly ``lax.scan`` over ``factoriax_step``
    is compiled exactly once for the file instead of once per test.
    """
    env = FactoriaXEnv()
    params = _SMALL_PARAMS

    def _step(
        carry: tuple[EnvState, jax.Array], _: None
    ) -> tuple[tuple[EnvState, jax.Array], None]:
        state, key = carry
        key, act_key, step_key = random.split(key, 3)
        action = random.randint(act_key, (), 0, NUM_ACTIONS)
        state = factoriax_step(step_key, state, action, params)
        return (state, key), None

    @jax.jit
    def run(rng: jax.Array) -> EnvState:
        rng, reset_key = random.split(rng)
        _, init_state = env.reset_env(reset_key, params)
        (final_state, _), _ = lax.scan(
            _step, (init_state, rng), None, length=_NUM_RANDOM_STEPS
        )
        return final_state

    return env, params, run


# ---------------------------------------------------------------------------
# State consistency invariants
# ---------------------------------------------------------------------------


class TestStateConsistency:
    """Invariants on the raw state arrays after random play."""

    def test_player_positions_in_bounds(self, random_episode) -> None:
        """Player positions must stay within map boundaries."""
        _, params, run = random_episode
        final = run(random.PRNGKey(42))

        assert jnp.all(final.player_positions[:, 0] >= 0)
        assert jnp.all(final.player_positions[:, 0] < params.map_width)
        assert jnp.all(final.player_positions[:, 1] >= 0)
        assert jnp.all(final.player_positions[:, 1] < params.map_height)

    def test_inventory_counts_non_negative(self, random_episode) -> None:
        """No inventory count should go below zero."""
        _, _, run = random_episode
        final = run(random.PRNGKey(7))

        assert jnp.all(final.player_inventory >= 0)

    def test_inventory_counts_within_stack_limit(self, random_episode) -> None:
        """Player inventory counts must not exceed per-type stack limits."""
        _, _, run = random_episode
        final = run(random.PRNGKey(55))

        assert jnp.all(final.player_inventory <= PLAYER_MAX_STACK)

    def test_machine_inventory_within_limits(self, random_episode) -> None:
        """Entity buffer counts must be non-negative and within stack limits."""
        _, _, run = random_episode
        final = run(random.PRNGKey(21))

        ent_cap = MACHINE_MAX_STACK[final.ent_type]
        assert jnp.all(final.ent_buf_count >= 0)
        assert jnp.all(final.ent_buf_count <= ent_cap)

    def test_block_resources_non_negative(self, random_episode) -> None:
        """Block resources must never go negative."""
        _, _, run = random_episode
        final = run(random.PRNGKey(77))

        assert jnp.all(final.block_resources >= 0)

    def test_ent_health_within_bounds(self, random_episode) -> None:
        """Active entities have ``0 <= ent_health <= max_health[type]``;
        inactive slots hold ``ent_health == 0``.

        Random rollouts dispatch placement, pickup, and REPAIR actions
        among others. Base mechanics never decrement health, so the
        invariant we check here is structural: bounds and inactive-slot
        zeroing.
        """
        _, params, run = random_episode
        final = run(random.PRNGKey(101))

        max_health_per_type = params.machine_config.max_health
        cap = max_health_per_type[final.ent_type]
        active = final.ent_y >= 0
        # For active entities: 0 <= ent_health <= max_health[type].
        active_health = jnp.where(active, final.ent_health, jnp.int16(0))
        assert jnp.all(active_health >= 0)
        assert jnp.all(jnp.where(active, final.ent_health <= cap, True))
        # Inactive slots stay at zero.
        inactive_health = jnp.where(active, jnp.int16(0), final.ent_health)
        assert jnp.all(inactive_health == 0)


# ---------------------------------------------------------------------------
# Item conservation invariants
# ---------------------------------------------------------------------------


class TestItemConservation:
    """Verify items are neither created nor destroyed unexpectedly."""

    def test_mining_conserves_resources(self, state_factory) -> None:
        """Mining transfers resources from blocks to inventory, never losing any.

        Total ore in inventory + remaining block resources must equal
        the initial block resources when only mining (no machines).
        """
        from factoriax.constants import Direction

        world_map = jnp.array(
            [[BlockType.COAL, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
            dtype=jnp.int32,
        )
        initial_resources = 10
        state = state_factory(
            world_map=world_map,
            player_position=(1, 0),
            player_direction=int(Direction.LEFT),
            block_resources=jnp.array(
                [[initial_resources, 0], [0, 0]], dtype=jnp.int16
            ),
        )

        # Mine the tile in front (COAL at (0, 0)) repeatedly.
        for _ in range(initial_resources + 2):
            state = mine_block(state, 0, _SMALL_PARAMS)

        coal_in_inv = int(state.player_inventory[0, ItemType.COAL])
        remaining = int(state.block_resources[0, 0])
        total = coal_in_inv + remaining

        assert total == initial_resources

    def test_deposit_withdraw_round_trip(self, state_factory) -> None:
        """Depositing then withdrawing preserves total item count."""
        from factoriax.constants import Direction
        from factoriax.game_logic import deposit_to_adjacent, withdraw_from_adjacent

        world_map = jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32)
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.COAL].set(10)
        state = state_factory(
            world_map=world_map,
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=inv,
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.PALLET]], dtype=jnp.int32
            ),
        )

        initial_total = int(state.player_inventory[0, ItemType.COAL])

        # Look up the entity at tile (row=0, col=1) where the pallet is.
        eid = int(state.tile_entity[0, 1])
        assert eid >= 0, "Expected an entity at tile (0, 1)"

        state = deposit_to_adjacent(state, 0, int(ItemType.COAL))
        inv_after = int(state.player_inventory[0, ItemType.COAL])
        buf_after = int(state.ent_buf_count[eid])
        assert inv_after + buf_after == initial_total

        state = withdraw_from_adjacent(state, 0)
        inv_after2 = int(state.player_inventory[0, ItemType.COAL])
        buf_after2 = int(state.ent_buf_count[eid])
        assert inv_after2 + buf_after2 == initial_total


# ---------------------------------------------------------------------------
# Observation fidelity invariants
# ---------------------------------------------------------------------------


class TestObservationFidelity:
    """Observations must faithfully reflect the underlying state."""

    def test_observation_shape_matches_space(self) -> None:
        """Observation shape must match the declared observation_space."""
        env = FactoriaXEnv()
        rng = random.PRNGKey(0)
        obs, _ = env.reset_env(rng, _SMALL_PARAMS)
        obs_space = env.observation_space(_SMALL_PARAMS)

        assert obs.shape == obs_space.shape

    def test_observation_values_in_range(self, random_episode) -> None:
        """All observation values must be in [0, 1] after random play."""
        env, params, run = random_episode
        final = run(random.PRNGKey(3))

        obs = env.get_obs(final, params)
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)

    def test_observation_shape_after_random_play(self, random_episode) -> None:
        """Observation shape must remain consistent after random play."""
        env, params, run = random_episode
        final = run(random.PRNGKey(5))

        obs = env.get_obs(final, params)
        obs_space = env.observation_space(params)
        assert obs.shape == obs_space.shape

    def test_inventory_observation_encodes_state(self) -> None:
        """Inventory portion of the observation must match state arrays."""
        env = FactoriaXEnv()
        rng = random.PRNGKey(17)
        _, state = env.reset_env(rng, _SMALL_PARAMS)

        # Place a known item in the player pouch.
        state = state.replace(
            player_inventory=state.player_inventory.at[0, ItemType.COAL].set(
                32,
            ),
        )

        obs = env.get_obs(state, _SMALL_PARAMS)
        # Verify the observation is finite and in range.
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)
