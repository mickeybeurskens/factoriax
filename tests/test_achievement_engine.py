"""Tests for the achievement-evaluation pass baked into FactoriaXEnv.step_env.

The achievement system used to live in a wrapper (AchievementWrapper);
this set of tests pins the engine-state version: a constructor argument
on FactoriaXEnv, evaluated and OR-folded inside step_env, with the
result available on state.achievements_unlocked. Spec: SPEC.md Phase A
items 1.2 and 1.3.

A module-scoped ``make_env`` factory caches ``(env, params, state)``
per achievement_fn so that ``factoriax_step``'s internal JIT cache is
shared across tests that use the same achievement_fn — each unique
achievement_fn compiles its step path at most once across the file.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from factoriax.achievements import ACHIEVEMENT_INFO, core_game_conditions
from factoriax.constants import MAX_ACHIEVEMENTS, NUM_ITEM_TYPES, BlockType, ItemType
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams, EnvState


def _achievement_index(achievement_id: str) -> int:
    """Look up the index of an achievement by its string id."""
    return next(i for i, a in enumerate(ACHIEVEMENT_INFO) if a.id == achievement_id)


def _all_true(state: EnvState) -> jnp.ndarray:
    """Achievement fn that unconditionally returns all-True."""
    return jnp.ones(MAX_ACHIEVEMENTS, dtype=jnp.bool_)


def _fire_when_timestep_one(state: EnvState) -> jnp.ndarray:
    """Achievement fn whose condition is True only when state.timestep == 1.

    Used to verify that latching outlives the condition: once the
    achievement fires at step 1, it must stay set even though the
    condition flips back to False at step 2.
    """
    one = jnp.ones(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    zero = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    return jnp.where(state.timestep == 1, one, zero)


@pytest.fixture(scope="module")
def make_env():
    """Module-scoped factory: achievement_fn -> (env, params, state).

    Cached on ``id(achievement_fn)`` so two tests passing the same
    callable share the same env instance, and the JIT cache that
    ``factoriax_step`` builds internally is reused across them.
    ``params`` is ``env.default_params`` (32x32, 2 players) and
    ``state`` is the post-reset state from ``random.PRNGKey(0)``.

    Tests that need a non-default shape build their own state and
    pass ``params`` of the right shape to ``env.step_env``; the env
    itself is reusable across params because ``achievement_fn`` is
    the only constructor-time concern.
    """
    cache: dict[int | None, tuple] = {}

    def _make(achievement_fn=None):
        key = id(achievement_fn) if achievement_fn is not None else None
        if key in cache:
            return cache[key]
        env = (
            FactoriaXEnv(achievement_fn=achievement_fn)
            if achievement_fn is not None
            else FactoriaXEnv()
        )
        params = env.default_params
        _, state = env.reset_env(random.PRNGKey(0), params)
        cache[key] = (env, params, state)
        return cache[key]

    return _make


def test_default_achievement_fn_keeps_unlocks_all_false(make_env) -> None:
    """No achievement_fn → achievements_unlocked stays all-False after a step."""
    env, params, state = make_env()
    _, state, _, _, _ = env.step_env(random.PRNGKey(1), state, 0, params)

    assert not bool(state.achievements_unlocked.any())


def test_engine_latches_unlocks_from_achievement_fn(make_env) -> None:
    """An achievement_fn that returns all-True unlocks every bit on step 1."""
    env, params, state = make_env(_all_true)

    assert not bool(state.achievements_unlocked.any())

    _, state, _, _, _ = env.step_env(random.PRNGKey(1), state, 0, params)

    assert bool(state.achievements_unlocked.all())


def test_unlocks_are_latched_across_steps(make_env) -> None:
    """Once an achievement unlocks, it stays unlocked when the condition flips off."""
    env, params, state = make_env(_fire_when_timestep_one)

    _, state, _, _, _ = env.step_env(random.PRNGKey(1), state, 0, params)
    assert bool(state.achievements_unlocked.all())  # condition fired at timestep=1

    _, state, _, _, _ = env.step_env(random.PRNGKey(2), state, 0, params)
    # condition is False at timestep=2 but unlocks must persist
    assert bool(state.achievements_unlocked.all())


def test_core_game_conditions_unlock_through_engine(make_env, state_factory) -> None:
    """End-to-end: core_game_conditions wired into the env latches first_ore.

    Uses a custom 1x1 state for this assertion; only the env is shared
    via ``make_env(core_game_conditions)`` so tests 5 and 6 in this file
    pick up the same env instance.
    """
    items_mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
    items_mined = items_mined.at[ItemType.IRON_ORE].set(1)
    state = state_factory(
        world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        items_mined=items_mined,
    )

    env, _, _ = make_env(core_game_conditions)
    params = EnvParams(map_width=1, map_height=1, num_players=1, max_machines=4)
    _, state, _, _, _ = env.step_env(random.PRNGKey(0), state, 0, params)

    assert bool(state.achievements_unlocked[_achievement_index("first_ore")])


def test_core_game_conditions_vmaps() -> None:
    """``core_game_conditions`` must vmap across batched envs.

    Uses 8x8 1p instead of the default 32x32 2p — the assertion is on
    ``achievements_unlocked.shape``, which depends on ``MAX_ACHIEVEMENTS``
    (not on map dimensions or player count), so the smaller shape gives
    identical coverage at a fraction of the vmap compile cost.

    A passing vmap also implies a passing plain-jit, so the previously
    separate ``test_core_game_conditions_jits_via_step`` was strictly
    subsumed and removed.
    """
    env = FactoriaXEnv(achievement_fn=core_game_conditions)
    params = EnvParams(map_width=8, map_height=8, num_players=1)

    reset_keys = random.split(random.PRNGKey(0), 4)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(reset_keys, params)

    step_keys = random.split(random.PRNGKey(1), 4)
    actions = jnp.zeros(4, dtype=jnp.int32)
    _, states, _, _, _ = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
        step_keys, states, actions, params
    )

    assert states.achievements_unlocked.shape == (4, MAX_ACHIEVEMENTS)
