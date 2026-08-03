"""Tests for the Mining-v1 scenario.

Scenario-level contract only. Mine-action mechanics live in
``test_mining.py``, observation encodings in
``test_observations_superficial.py``, and wrapper semantics in
``test_env_contract.py`` / ``test_env_hooks.py``. Covered here:
terrain layout, the scenario reward's bookkeeping over an episode,
termination, the MLP-friendly default obs, and a scripted-oracle
solvability regression (max score 30 reachable within the step budget).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import random

from factoriax.engine.constants import Action, BlockType, Direction, ItemType
from factoriax.engine.envs.mining import mining
from factoriax.engine.tables import DIRECTIONS
from factoriax.make import env_from_name

MAP_SIZE = 8
MAX_SCORE = 30

_DIR_TO_MOVE = {
    Direction.UP: Action.UP,
    Direction.DOWN: Action.DOWN,
    Direction.LEFT: Action.LEFT,
    Direction.RIGHT: Action.RIGHT,
}
_DIR_TO_FACE = {
    Direction.UP: Action.FACE_UP,
    Direction.DOWN: Action.FACE_DOWN,
    Direction.LEFT: Action.FACE_LEFT,
    Direction.RIGHT: Action.FACE_RIGHT,
}


@pytest.fixture(scope="module")
def env_and_params():
    return mining()


@pytest.fixture(scope="module")
def jit_step(env_and_params):
    env, _ = env_and_params
    return jax.jit(env.step_env)


def test_default_obs_is_local_mlp_friendly(env_and_params) -> None:
    """Mining-v1 is an RL baseline: egocentric obs, full-map radius."""
    env, params = env_and_params
    assert env.obs == "superficial_local"
    assert env.obs_radius == MAP_SIZE - 1
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape


def test_terrain_has_ten_ore_tiles_of_three(env_and_params) -> None:
    env, params = env_and_params
    for seed in range(5):
        _, state = env.reset_env(random.PRNGKey(seed), params)
        m = np.asarray(state.map)
        ore = m == int(BlockType.IRON)
        assert ore.sum() == 10
        assert not ore[MAP_SIZE // 2, MAP_SIZE // 2]
        assert (np.asarray(state.block_resources)[ore] == 3).all()
        assert (np.asarray(state.block_resources)[~ore] == 0).all()


def test_episode_reward_accounting_and_termination(env_and_params, jit_step) -> None:
    """sum(rewards) == items_mined == inventory. Done fires only at t=100."""
    env, params = env_and_params
    key = random.PRNGKey(7)
    _, state = env.reset_env(key, params)

    total = 0.0
    dones = []
    for _ in range(int(params.max_timesteps)):
        key, ka, ks = random.split(key, 3)
        action = int(random.randint(ka, (), 0, 10))
        _, state, reward, done, _ = jit_step(ks, state, action, params)
        total += float(reward)
        dones.append(bool(done))

    assert total == int(state.items_mined[ItemType.IRON_ORE])
    assert total == int(state.player_inventory[0, ItemType.IRON_ORE])
    assert not any(dones[:-1])
    assert dones[-1]


def _oracle_action(state) -> int:
    """Greedy scripted miner: face and mine adjacent ore, else approach."""
    m = np.asarray(state.map)
    px, py = (int(v) for v in state.player_positions[0])
    ys, xs = np.where(m == int(BlockType.IRON))
    if len(xs) == 0:
        return int(Action.NOOP)

    for d in _DIR_TO_MOVE:
        off = np.asarray(DIRECTIONS[int(d)])
        tx, ty = px + int(off[0]), py + int(off[1])
        if 0 <= tx < MAP_SIZE and 0 <= ty < MAP_SIZE:
            if m[ty, tx] == int(BlockType.IRON):
                if int(state.player_directions[0]) == int(d):
                    return int(Action.MINE)
                return int(_DIR_TO_FACE[d])

    dists = np.abs(xs - px) + np.abs(ys - py)
    i = int(np.argmin(dists))
    dx, dy = int(xs[i]) - px, int(ys[i]) - py
    if abs(dx) >= abs(dy):
        return int(_DIR_TO_MOVE[Direction.RIGHT if dx > 0 else Direction.LEFT])
    return int(_DIR_TO_MOVE[Direction.DOWN if dy > 0 else Direction.UP])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_oracle_reaches_max_score(env_and_params, jit_step, seed) -> None:
    """A greedy scripted policy collects all 30 ore within the 100-step
    budget. This guards against mechanics or generator changes that
    silently make the benchmark unsolvable."""
    env, params = env_and_params
    key = random.PRNGKey(seed)
    _, state = env.reset_env(key, params)

    total = 0.0
    for _ in range(int(params.max_timesteps)):
        action = _oracle_action(state)
        key, ks = random.split(key)
        _, state, reward, _, _ = jit_step(ks, state, action, params)
        total += float(reward)
        if total >= MAX_SCORE:
            break

    assert total == MAX_SCORE


def test_registry_default_matches_factory() -> None:
    env, _ = env_from_name("Mining-v1")
    assert env.obs == "superficial_local"
    assert env.obs_radius == MAP_SIZE - 1
