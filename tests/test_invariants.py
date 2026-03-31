"""Property-based invariant tests for the FactoriaX environment.

These tests verify structural invariants that must hold after any
sequence of actions, regardless of the specific actions taken. They
catch subtle bugs that targeted unit tests miss by exercising the
environment with random action sequences.
"""

import jax
import jax.numpy as jnp
from jax import lax, random

from factoriax.constants import (
    MAX_MACHINE_STACK_SIZE,
    MAX_STACK_SIZE,
    NUM_ACTIONS,
    NUM_INVENTORY_SLOTS,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.game_logic import factoriax_step, mine_block
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.recipes import MAX_ASSEMBLER_STACK_SIZE
from factoriax.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_PARAMS = EnvParams(map_width=8, map_height=8, max_timesteps=200)
_NUM_RANDOM_STEPS = 100


def _run_random_episode(
    rng: jax.Array,
    env: FactoriaXEnv,
    params: EnvParams,
    num_steps: int,
) -> tuple[EnvState, EnvState]:
    """Run *num_steps* random actions and return (initial, final) states.

    Args:
        rng: JAX random key.
        env: FactoriaX environment instance.
        params: Environment parameters.
        num_steps: Number of steps to execute.

    Returns:
        Tuple of (initial_state, final_state).
    """
    rng, reset_key = random.split(rng)
    _, init_state = env.reset_env(reset_key, params)

    def step_fn(
        carry: tuple[EnvState, jax.Array], _: None
    ) -> tuple[tuple[EnvState, jax.Array], None]:
        state, key = carry
        key, act_key, step_key = random.split(key, 3)
        action = random.randint(act_key, (), 0, NUM_ACTIONS)
        state = factoriax_step(step_key, state, action, params)
        return (state, key), None

    (final_state, _), _ = lax.scan(
        step_fn, (init_state, rng), None, length=num_steps
    )
    return init_state, final_state


# ---------------------------------------------------------------------------
# State consistency invariants
# ---------------------------------------------------------------------------


class TestStateConsistency:
    """Invariants on the raw state arrays after random play."""

    def test_player_positions_in_bounds(self) -> None:
        """Player positions must stay within map boundaries."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(42), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        assert jnp.all(final.player_positions[:, 0] >= 0)
        assert jnp.all(final.player_positions[:, 0] < _SMALL_PARAMS.map_width)
        assert jnp.all(final.player_positions[:, 1] >= 0)
        assert jnp.all(final.player_positions[:, 1] < _SMALL_PARAMS.map_height)

    def test_inventory_counts_non_negative(self) -> None:
        """No inventory slot count should go below zero."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(7), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        assert jnp.all(final.inventory_counts >= 0)

    def test_inventory_slot_consistency(self) -> None:
        """EMPTY item type iff count is zero; non-EMPTY iff count > 0."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(13), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        empty_items = final.inventory_items == ItemType.EMPTY
        zero_counts = final.inventory_counts == 0
        assert jnp.all(empty_items == zero_counts)

    def test_selected_slot_in_range(self) -> None:
        """Selected slot index stays within [0, NUM_INVENTORY_SLOTS)."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(99), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        assert jnp.all(final.selected_slots >= 0)
        assert jnp.all(final.selected_slots < NUM_INVENTORY_SLOTS)

    def test_machine_inventory_within_limits(self) -> None:
        """Machine slot counts must not exceed the stack limit."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(21), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        is_asm = final.machine_types == MachineType.ASSEMBLER
        asm_cap = jnp.where(
            is_asm[:, :, None],
            MAX_ASSEMBLER_STACK_SIZE,
            MAX_MACHINE_STACK_SIZE,
        )
        assert jnp.all(final.machine_inventory_counts >= 0)
        assert jnp.all(final.machine_inventory_counts <= asm_cap)

    def test_inventory_counts_within_stack_limit(self) -> None:
        """Player inventory counts must not exceed MAX_STACK_SIZE."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(55), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        assert jnp.all(final.inventory_counts <= MAX_STACK_SIZE)

    def test_block_resources_non_negative(self) -> None:
        """Block resources must never go negative."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(77), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        assert jnp.all(final.block_resources >= 0)


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
        from factoriax.constants import BlockType

        world_map = jnp.array(
            [[BlockType.COAL, BlockType.DIRT],
             [BlockType.DIRT, BlockType.DIRT]],
            dtype=jnp.int32,
        )
        initial_resources = 10
        state = state_factory(
            world_map=world_map,
            player_position=(0, 0),
            block_resources=jnp.array(
                [[initial_resources, 0], [0, 0]], dtype=jnp.int16
            ),
        )

        # Mine repeatedly.
        for _ in range(initial_resources + 2):
            state = mine_block(state, 0)

        coal_in_inv = jnp.sum(
            jnp.where(
                state.inventory_items[0] == ItemType.COAL,
                state.inventory_counts[0],
                0,
            )
        )
        remaining = state.block_resources[0, 0]
        total = int(coal_in_inv) + int(remaining)

        assert total == initial_resources

    def test_deposit_withdraw_round_trip(self, state_factory) -> None:
        """Depositing then withdrawing preserves total item count."""
        from factoriax.constants import Action
        from factoriax.game_logic import deposit_to_adjacent, withdraw_from_adjacent

        world_map = jnp.array(
            [[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32
        )
        state = state_factory(
            world_map=world_map,
            player_position=(0, 0),
            player_direction=int(Action.RIGHT),
            inventory_items=jnp.array([[ItemType.COAL] + [0] * 9], dtype=jnp.int32),
            inventory_counts=jnp.array([[10] + [0] * 9], dtype=jnp.int32),
            machine_types=jnp.array(
                [[MachineType.NONE, MachineType.CHEST]], dtype=jnp.int32
            ),
        )

        initial_total = int(jnp.sum(state.inventory_counts))

        state = deposit_to_adjacent(state, 0)
        inv_after_deposit = int(jnp.sum(state.inventory_counts))
        machine_after_deposit = int(
            jnp.sum(state.machine_inventory_counts[0, 1])
        )
        assert inv_after_deposit + machine_after_deposit == initial_total

        state = withdraw_from_adjacent(state, 0)
        inv_after_withdraw = int(jnp.sum(state.inventory_counts))
        machine_after_withdraw = int(
            jnp.sum(state.machine_inventory_counts[0, 1])
        )
        assert inv_after_withdraw + machine_after_withdraw == initial_total


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

    def test_observation_values_in_range(self) -> None:
        """All observation values must be in [0, 1] after random play."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(3), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        obs = env.get_obs(final, _SMALL_PARAMS)
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)

    def test_observation_shape_after_random_play(self) -> None:
        """Observation shape must remain consistent after random play."""
        env = FactoriaXEnv()
        _, final = _run_random_episode(
            random.PRNGKey(5), env, _SMALL_PARAMS, _NUM_RANDOM_STEPS
        )

        obs = env.get_obs(final, _SMALL_PARAMS)
        obs_space = env.observation_space(_SMALL_PARAMS)
        assert obs.shape == obs_space.shape

    def test_inventory_observation_encodes_state(self) -> None:
        """Inventory portion of the observation must match state arrays."""
        env = FactoriaXEnv()
        rng = random.PRNGKey(17)
        _, state = env.reset_env(rng, _SMALL_PARAMS)

        # Place a known item in the inventory.
        state = state.replace(
            inventory_items=state.inventory_items.at[0, 0].set(ItemType.COAL),
            inventory_counts=state.inventory_counts.at[0, 0].set(32),
        )

        obs = env.get_obs(state, _SMALL_PARAMS)
        spatial_size = (
            NUM_SPATIAL_CHANNELS
            * _SMALL_PARAMS.map_width
            * _SMALL_PARAMS.map_height
        )
        inv_items_start = spatial_size + NUM_PLAYER_SCALARS
        inv_counts_start = inv_items_start + NUM_INVENTORY_SLOTS

        from factoriax.constants import NUM_ITEM_TYPES

        expected_item = float(ItemType.COAL) / NUM_ITEM_TYPES
        expected_count = 32.0 / MAX_STACK_SIZE

        assert abs(float(obs[inv_items_start]) - expected_item) < 1e-5
        assert abs(float(obs[inv_counts_start]) - expected_count) < 1e-5
