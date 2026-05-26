"""Tests for ``factoriax.envs.science_tally_wrapper.ScienceTallyWrapper``.

The wrapper's contract is narrow: forward to the inner env's step, then
fold ``new_env_state.science_consumed_step`` into ``total_science_consumed``.
``run_labs`` is what writes the per-step delta in the first place; that
behaviour is exercised in ``tests/test_science_lab.py`` against the real
engine. Here, a ``_StubInner`` returns the next state directly so the
wrapper's accumulation logic can be verified without paying the
``factoriax_step`` XLA compile.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from factoriax.engine.constants import NUM_SCIENCE_PACK_TYPES
from factoriax.engine.state import EnvParams
from factoriax.envs.science_tally_wrapper import (
    ScienceTallyState,
    ScienceTallyWrapper,
)


def _base_params() -> EnvParams:
    return EnvParams(map_width=8, map_height=8, num_players=1)


class _StubInner:
    """Stand-in inner env: passes state through unchanged on each step.

    The wrapper reads ``new_env_state.science_consumed_step`` after the
    inner step. By returning the same state the test passed in, we let
    the test pre-set the delta and assert the wrapper's accumulation
    against it. Implements only what ``ScienceTallyWrapper`` reaches for.
    """

    default_params = None

    def step_env(
        self,
        key: jax.Array,
        state: Any,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]]:
        return jnp.zeros(1), state, jnp.float32(0.0), jnp.bool_(False), {}

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, Any]:
        # Wrapper.reset_env wraps the returned state in a ScienceTallyState
        # with a zero tally; the inner state's content is irrelevant here.
        return jnp.zeros(1), object()

    def get_obs(self, state: Any, params: EnvParams) -> jax.Array:
        return jnp.zeros(3)

    def is_terminal(self, state: Any, params: EnvParams) -> jax.Array:
        return jnp.bool_(False)

    def action_space(self, params: EnvParams) -> Any:
        return None

    def observation_space(self, params: EnvParams) -> Any:
        return None


@pytest.fixture(scope="module")
def tally_env() -> ScienceTallyWrapper:
    """``ScienceTallyWrapper`` wrapping a stub inner — no compile."""
    return ScienceTallyWrapper(_StubInner())  # type: ignore[arg-type]


def _state_with_delta(state_factory, delta: tuple[int, int]):
    """Build an EnvState whose ``science_consumed_step`` is set to ``delta``."""
    return state_factory(
        world_map=jnp.zeros((1, 1), dtype=jnp.int32),
        science_consumed_step=jnp.array(delta, dtype=jnp.int32),
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
        """One step folds ``science_consumed_step`` into the total."""
        inner_state = _state_with_delta(state_factory, (5, 3))
        wrapped = ScienceTallyState(
            env_state=inner_state,
            total_science_consumed=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        )
        _, wrapped, _, _, _ = tally_env.step_env(
            jax.random.PRNGKey(1), wrapped, 0, _base_params()
        )
        assert tuple(wrapped.total_science_consumed.tolist()) == (5, 3)

    def test_tally_accumulates_across_steps(self, tally_env, state_factory) -> None:
        """Three consecutive deltas accumulate correctly."""
        total = jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32)
        step_deltas = [(4, 0), (0, 3), (2, 2)]
        expected_running = [(4, 0), (4, 3), (6, 5)]

        for delta, expected in zip(step_deltas, expected_running, strict=True):
            inner_state = _state_with_delta(state_factory, delta)
            wrapped = ScienceTallyState(
                env_state=inner_state, total_science_consumed=total
            )
            _, wrapped, _, _, _ = tally_env.step_env(
                jax.random.PRNGKey(42), wrapped, 0, _base_params()
            )
            total = wrapped.total_science_consumed
            assert tuple(total.tolist()) == expected

    def test_reset_clears_prior_tally(self, tally_env, state_factory) -> None:
        """``reset_env`` zeroes the tally even after prior consumption."""
        inner_state = _state_with_delta(state_factory, (8, 0))
        wrapped = ScienceTallyState(
            env_state=inner_state,
            total_science_consumed=jnp.array([100, 50], dtype=jnp.int32),
        )
        _, wrapped, _, _, _ = tally_env.step_env(
            jax.random.PRNGKey(0), wrapped, 0, _base_params()
        )
        assert tuple(wrapped.total_science_consumed.tolist()) == (108, 50)

        _, wrapped_after_reset = tally_env.reset_env(
            jax.random.PRNGKey(0), _base_params()
        )
        assert tuple(wrapped_after_reset.total_science_consumed.tolist()) == (0, 0)

    def test_obs_passthrough(self, tally_env, state_factory) -> None:
        """``get_obs`` returns exactly what the inner env returns."""
        inner_state = _state_with_delta(state_factory, (0, 0))
        wrapped = ScienceTallyState(
            env_state=inner_state,
            total_science_consumed=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        )
        obs_inner = tally_env._inner.get_obs(inner_state, _base_params())
        obs_wrap = tally_env.get_obs(wrapped, _base_params())
        assert jnp.array_equal(obs_inner, obs_wrap)
