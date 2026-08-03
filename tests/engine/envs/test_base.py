"""Tests for the extension points on :class:`FactoriaxEnv`.

``engine/envs/base.py`` takes a scenario's behaviour as constructor
arguments rather than through a subclass: ``terrain_fn``, the step and
reset hooks, ``reward_fn``, ``done_fn``, and the achievement hook. Each one
is additive, so an env that passes none of them behaves as the plain
engine.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from factoriax.engine.constants import MAX_ACHIEVEMENTS, Action
from factoriax.engine.envs.base import FactoriaxEnv, achievement_hook
from factoriax.engine.levels import build_state

_NOOP = int(Action.NOOP)


def _bit(index: int) -> jnp.ndarray:
    return jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_).at[index].set(True)


def test_terrain_fn_overrides_level(level8, params) -> None:
    """A supplied ``terrain_fn`` is used and the bound ``level`` is ignored."""
    from factoriax.engine.constants import BlockType

    def all_dirt(key, p):
        del key, p
        return jnp.full((8, 8), int(BlockType.DIRT), dtype=jnp.int32)

    env = FactoriaxEnv(terrain_fn=all_dirt, level=level8)
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert jnp.all(state.map == int(BlockType.DIRT))


def test_step_hooks_run_after_step(params) -> None:
    """Step hooks are applied after the physics step, in order."""

    def set_bit0(key, state, p):
        del key, p
        return state.replace(
            achievements_unlocked=state.achievements_unlocked.at[0].set(True)
        )

    env = FactoriaxEnv(step_hooks=(set_bit0,))
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert not bool(state.achievements_unlocked[0])
    _, new_state, _, _, _ = env.step_env(random.PRNGKey(1), state, _NOOP, params)
    assert bool(new_state.achievements_unlocked[0])


def test_reset_hooks_transform_initial_state(level8, params) -> None:
    """Reset hooks run after state construction on every reset path."""
    from factoriax.engine.constants import ItemType

    def stock_miners(key, state, p):
        del key, p
        return state.replace(
            player_inventory=state.player_inventory.at[:, int(ItemType.MINER)].set(6)
        )

    # Procedural path.
    env = FactoriaxEnv(map_width=8, map_height=8, reset_hooks=(stock_miners,))
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert int(state.player_inventory[0, int(ItemType.MINER)]) == 6

    # Level-bound path.
    env_level = FactoriaxEnv(level=level8, reset_hooks=(stock_miners,))
    _, state_level = env_level.reset_env(random.PRNGKey(0), params)
    assert int(state_level.player_inventory[0, int(ItemType.MINER)]) == 6

    # Default: no hooks, inventory stays empty.
    env_plain = FactoriaxEnv(map_width=8, map_height=8)
    _, state_plain = env_plain.reset_env(random.PRNGKey(0), params)
    assert not bool(jnp.any(state_plain.player_inventory))


def test_reward_fn_used_else_zero(params) -> None:
    """``step_env`` returns ``reward_fn``'s value, or 0.0 when unset."""
    env_r = FactoriaxEnv(reward_fn=lambda prev, new, p: jnp.float32(7.0))
    _, state = env_r.reset_env(random.PRNGKey(0), params)
    _, _, reward, _, _ = env_r.step_env(random.PRNGKey(1), state, _NOOP, params)
    assert float(reward) == 7.0

    env_0 = FactoriaxEnv()
    _, state0 = env_0.reset_env(random.PRNGKey(0), params)
    _, _, reward0, _, _ = env_0.step_env(random.PRNGKey(1), state0, _NOOP, params)
    assert float(reward0) == 0.0


def test_done_fn_ors_with_timestep_limit(params) -> None:
    """``done_fn`` can end the episode before max_timesteps. The default is
    unchanged."""

    def after_one_step(state, p):
        del p
        return state.timestep >= 1

    env = FactoriaxEnv(map_width=8, map_height=8, done_fn=after_one_step)
    _, state = env.reset_env(random.PRNGKey(0), params)
    _, state, _, done, _ = env.step_env(random.PRNGKey(1), state, _NOOP, params)
    assert bool(done)
    assert bool(env.is_terminal(state, params))

    env_plain = FactoriaxEnv(map_width=8, map_height=8)
    _, state0 = env_plain.reset_env(random.PRNGKey(0), params)
    _, state0, _, done0, _ = env_plain.step_env(
        random.PRNGKey(1), state0, _NOOP, params
    )
    assert not bool(done0)
    assert not bool(env_plain.is_terminal(state0, params))


def test_achievement_hook_latches(level8, params) -> None:
    """``achievement_hook`` OR-folds bits and they latch across applications."""
    hook_a = achievement_hook(lambda s: _bit(3))
    hook_b = achievement_hook(lambda s: _bit(5))
    state = build_state(level8, num_players=1)
    assert not bool(jnp.any(state.achievements_unlocked))
    state = hook_a(random.PRNGKey(0), state, params)
    assert bool(state.achievements_unlocked[3])
    state = hook_b(random.PRNGKey(0), state, params)
    assert bool(state.achievements_unlocked[3]) and bool(state.achievements_unlocked[5])
