"""Throwaway A/B harness for ``SPEC_TEST_SUITE.md`` Phase 1.

This file is **not** a regression test — it lives outside ``tests/``
on purpose. Its job is to A/B the cost of sharing a JITted
``env.step_env`` across tests at session scope versus rebuilding it
per test. Two classes run *the same* ten assertions:

- :class:`TestBaselineFunctionScope` — each test builds its own
  :class:`FactoriaXEnv`, wraps ``env.step_env`` with :func:`jax.jit`
  locally, and pays a fresh XLA compile every time. Lands in Task 1.1.
- :class:`TestSharedSessionScope` — each test consumes a session-scoped
  fixture so the compile is paid once for the whole class. Lands in
  Task 1.2.

The assertions are deliberately cheap (shape, dtype, cache size,
timestep) so the dominant cost is JIT compile, not Python work. The
side-by-side timing of the two classes is the experiment's signal.

This entire directory gets deleted once the gate question is
answered. See ``tasks/test_suite_todo.md`` cleanup step.
"""

from __future__ import annotations

import jax
from jax import random

from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams


def _build_env_and_state() -> tuple[FactoriaXEnv, EnvParams, object, jax.Array]:
    """Construct a fresh env + reset state at the canonical 8x8 1p shape.

    Returns:
        Tuple ``(env, params, initial_state, rng)``. ``rng`` is the
        post-reset key, ready to be split for a step call.
    """
    env = FactoriaXEnv()
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    rng = random.PRNGKey(0)
    rng, reset_key = random.split(rng)
    _, state = env.reset_env(reset_key, params)
    return env, params, state, rng


class TestBaselineFunctionScope:
    """Per-test JIT baseline — each method rebuilds env + step_fn.

    Every method pays the full XLA compile cost. The ten assertions
    below match :class:`TestSharedSessionScope` line for line; the only
    delta between the two classes is *where* the env is built.
    """

    def test_step_returns_five_tuple(self) -> None:
        """``env.step_env`` returns ``(obs, state, reward, done, info)``."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        result = step_fn(step_key, state, int(Action.NOOP), params)
        assert len(result) == 5

    def test_timestep_increments_by_one(self) -> None:
        """A single NOOP step advances ``state.timestep`` from 0 to 1."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        assert int(state.timestep) == 0
        rng, step_key = random.split(rng)
        _, new_state, _, _, _ = step_fn(step_key, state, int(Action.NOOP), params)
        assert int(new_state.timestep) == 1

    def test_obs_shape_non_empty(self) -> None:
        """The observation vector has at least one element."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        obs, _, _, _, _ = step_fn(step_key, state, int(Action.NOOP), params)
        assert obs.shape[0] > 0

    def test_reward_is_finite_float(self) -> None:
        """Reward is castable to a finite Python float."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        _, _, reward, _, _ = step_fn(step_key, state, int(Action.NOOP), params)
        r = float(reward)
        assert r == r  # NaN check via self-equality

    def test_done_is_bool_scalar(self) -> None:
        """``done`` round-trips through ``bool()`` without raising."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        _, _, _, done, _ = step_fn(step_key, state, int(Action.NOOP), params)
        assert isinstance(bool(done), bool)

    def test_state_pytree_structure_preserved(self) -> None:
        """Reset state and post-step state share the same pytree structure."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        _, new_state, _, _, _ = step_fn(step_key, state, int(Action.NOOP), params)
        _, treedef_before = jax.tree.flatten(state)
        _, treedef_after = jax.tree.flatten(new_state)
        assert treedef_before == treedef_after

    def test_state_leaves_match_shapes(self) -> None:
        """Every array leaf retains its shape across one step."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        _, new_state, _, _, _ = step_fn(step_key, state, int(Action.NOOP), params)
        leaves_before, _ = jax.tree.flatten(state)
        leaves_after, _ = jax.tree.flatten(new_state)
        for before, after in zip(leaves_before, leaves_after, strict=True):
            if hasattr(before, "shape"):
                assert before.shape == after.shape

    def test_state_leaves_match_dtypes(self) -> None:
        """Every array leaf retains its dtype across one step."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        _, new_state, _, _, _ = step_fn(step_key, state, int(Action.NOOP), params)
        leaves_before, _ = jax.tree.flatten(state)
        leaves_after, _ = jax.tree.flatten(new_state)
        for before, after in zip(leaves_before, leaves_after, strict=True):
            if hasattr(before, "dtype"):
                assert before.dtype == after.dtype

    def test_jit_cache_size_is_one_after_step(self) -> None:
        """One step against a fresh ``jax.jit`` wrapper caches one trace."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        step_fn(step_key, state, int(Action.NOOP), params)
        assert step_fn._cache_size() == 1

    def test_repeated_step_does_not_retrace(self) -> None:
        """Two NOOP steps reuse the cached trace — cache size stays one."""
        env, params, state, rng = _build_env_and_state()
        step_fn = jax.jit(env.step_env)
        rng, step_key = random.split(rng)
        _, state, _, _, _ = step_fn(step_key, state, int(Action.NOOP), params)
        rng, step_key = random.split(rng)
        step_fn(step_key, state, int(Action.NOOP), params)
        assert step_fn._cache_size() == 1
