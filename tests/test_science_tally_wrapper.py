"""Tests for ``factoriax.envs.science_tally_wrapper.ScienceTallyWrapper``.

Verifies:

1. ``reset_env`` zeroes the tally.
2. ``step_env`` accumulates the per-step delta into the total.
3. A second ``reset_env`` after consumption clears the tally again.
4. The wrapper is observationally transparent — ``get_obs`` returns the
   same vector as the inner env's ``get_obs`` for the same state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from factoriax.constants import (
    NUM_SCIENCE_PACK_TYPES,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.envs import FactoriaXEnv
from factoriax.envs.science_tally_wrapper import (
    ScienceTallyState,
    ScienceTallyWrapper,
)
from factoriax.state import EnvParams


def _base_params() -> EnvParams:
    return EnvParams(map_width=8, map_height=8, num_players=1)


@pytest.fixture(scope="module")
def tally_env() -> ScienceTallyWrapper:
    """Module-scoped ``ScienceTallyWrapper(FactoriaXEnv())`` at 8x8 1p.

    All six tests in this file wrap the default env at the same shape;
    sharing the wrapper instance lets ``factoriax_step``'s internal
    JIT cache amortise across them instead of compiling per test.
    """
    return ScienceTallyWrapper(FactoriaXEnv())


def _lab_state_from_factory(
    state_factory,
    types: tuple[int, int],
    counts: tuple[int, int],
) -> object:
    """Build an EnvState with a single loaded SCIENCE_LAB."""
    h = w = 8
    world_map = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
    mt_grid = (
        jnp.full((h, w), int(MachineType.NONE), dtype=jnp.int32)
        .at[3, 3]
        .set(int(MachineType.SCIENCE_LAB))
    )
    ait = (
        jnp.zeros((h, w, 2), dtype=jnp.int32)
        .at[3, 3, 0]
        .set(types[0])
        .at[3, 3, 1]
        .set(types[1])
    )
    aic = (
        jnp.zeros((h, w, 2), dtype=jnp.int32)
        .at[3, 3, 0]
        .set(counts[0])
        .at[3, 3, 1]
        .set(counts[1])
    )
    return state_factory(
        world_map=world_map,
        machine_types=mt_grid,
        asm_in_type=ait,
        asm_in_count=aic,
    )


class TestScienceTallyWrapper:
    """End-to-end behaviour of the wrapper."""

    def test_tally_zero_at_reset(self, tally_env) -> None:
        _, wrapped = tally_env.reset_env(jax.random.PRNGKey(0), _base_params())
        assert isinstance(wrapped, ScienceTallyState)
        assert tuple(wrapped.total_science_consumed.tolist()) == (0, 0)

    def test_tally_dtype_and_shape(self, tally_env) -> None:
        _, wrapped = tally_env.reset_env(jax.random.PRNGKey(0), _base_params())
        assert wrapped.total_science_consumed.shape == (NUM_SCIENCE_PACK_TYPES,)
        assert wrapped.total_science_consumed.dtype == jnp.int32

    def test_tally_accumulates_single_step(self, tally_env, state_factory) -> None:
        """One step with a loaded lab adds its delta to the total."""
        params = _base_params()
        inner_state = _lab_state_from_factory(
            state_factory,
            types=(
                int(ItemType.BASIC_SCIENCE_PACK),
                int(ItemType.ADVANCED_SCIENCE_PACK),
            ),
            counts=(5, 3),
        )
        wrapped = ScienceTallyState(
            env_state=inner_state,
            total_science_consumed=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        )
        _, wrapped, _, _, _ = tally_env.step_env(
            jax.random.PRNGKey(1), wrapped, 0, params
        )
        assert tuple(wrapped.total_science_consumed.tolist()) == (5, 3)

    def test_tally_accumulates_across_steps(self, tally_env, state_factory) -> None:
        """Three consecutive consumption steps accumulate correctly."""
        params = _base_params()

        total = jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32)
        expected_running = [(0, 0), (4, 0), (4, 3), (6, 5)]
        step_schedule = [
            (int(ItemType.BASIC_SCIENCE_PACK), 0, 4, 0),
            (0, int(ItemType.ADVANCED_SCIENCE_PACK), 0, 3),
            (
                int(ItemType.BASIC_SCIENCE_PACK),
                int(ItemType.ADVANCED_SCIENCE_PACK),
                2,
                2,
            ),
        ]

        # Start totals match the first entry of expected_running.
        assert tuple(total.tolist()) == expected_running[0]

        for (t0, t1, c0, c1), expected in zip(
            step_schedule, expected_running[1:], strict=True
        ):
            inner_state = _lab_state_from_factory(
                state_factory, types=(t0, t1), counts=(c0, c1)
            )
            wrapped = ScienceTallyState(
                env_state=inner_state, total_science_consumed=total
            )
            _, wrapped, _, _, _ = tally_env.step_env(
                jax.random.PRNGKey(42), wrapped, 0, params
            )
            total = wrapped.total_science_consumed
            assert tuple(total.tolist()) == expected

    def test_reset_clears_prior_tally(self, tally_env, state_factory) -> None:
        """``reset_env`` zeroes the tally even after prior consumption."""
        params = _base_params()
        inner_state = _lab_state_from_factory(
            state_factory,
            types=(int(ItemType.BASIC_SCIENCE_PACK), 0),
            counts=(8, 0),
        )
        wrapped = ScienceTallyState(
            env_state=inner_state,
            total_science_consumed=jnp.array([100, 50], dtype=jnp.int32),
        )
        _, wrapped, _, _, _ = tally_env.step_env(
            jax.random.PRNGKey(0), wrapped, 0, params
        )
        assert tuple(wrapped.total_science_consumed.tolist()) == (108, 50)

        _, wrapped_after_reset = tally_env.reset_env(jax.random.PRNGKey(0), params)
        assert tuple(wrapped_after_reset.total_science_consumed.tolist()) == (0, 0)

    def test_obs_passthrough(self, tally_env, canonical_env_8x8_1p) -> None:
        """Wrapped ``get_obs`` returns the same array as the inner env's."""
        inner_env, params, _, inner_state = canonical_env_8x8_1p
        wrapped = ScienceTallyState(
            env_state=inner_state,
            total_science_consumed=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        )
        obs_inner = inner_env.get_obs(inner_state, params)
        obs_wrap = tally_env.get_obs(wrapped, params)
        assert jnp.array_equal(obs_inner, obs_wrap)
