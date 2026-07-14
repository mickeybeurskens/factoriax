"""Tests for the MinerBootstrap-v1 scenario.

Scenario-level contracts for the sparse miner-bootstrap task (see the
module docstring of ``factoriax.engine.envs.miner_curriculum``).
Covered: registration, the transfer invariant (obs shape identical to
EasyRocket-v1 built with the same obs variant), start-state contract,
completion-only reward accounting (1.0 exactly once, on the completing
step; every intermediate event pays 0; timeout pays 0; the 14 ladder
bits still latch as free diagnostics), and a scripted-oracle
solvability check within the 300-step budget.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import random

from factoriax.engine.constants import (
    Action,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.envs.easy_rocket import easy_rocket
from factoriax.engine.envs.miner_curriculum import (
    MINER_BOOTSTRAP_MAX_SCORE,
    NUM_MINER_BOOTSTRAP_ACHIEVEMENTS,
    miner_bootstrap,
)
from factoriax.engine.tables import DIRECTIONS
from factoriax.make import env_from_name
from tests.scenarios.oracle_utils import (
    BLOCK_TO_ITEM as _BLOCK_TO_ITEM,
)
from tests.scenarios.oracle_utils import (
    DIR_TO_FACE as _DIR_TO_FACE,
)
from tests.scenarios.oracle_utils import (
    DIR_TO_MOVE as _DIR_TO_MOVE,
)
from tests.scenarios.oracle_utils import (
    bfs_step_toward as _bfs_step_toward,
)
from tests.scenarios.oracle_utils import (
    face_or_mine_adjacent as _face_or_mine_adjacent,
)

MAX_TIMESTEPS = 300


@pytest.fixture(scope="module")
def bootstrap_env():
    return miner_bootstrap()


@pytest.fixture(scope="module")
def bootstrap_step(bootstrap_env):
    env, _ = bootstrap_env
    return jax.jit(env.step_env)


def test_bootstrap_registered() -> None:
    env, params = env_from_name("MinerBootstrap-v1")
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape


def test_bootstrap_obs_shape_matches_easy_rocket(bootstrap_env) -> None:
    """The transfer invariant: policy weights must load on EasyRocket-v1
    built with the scenario's obs variant (the scenario defaults to
    ``superficial_local``; EasyRocket-v1's own default stays global)."""
    env, params = bootstrap_env
    er_env, er_params = easy_rocket(obs=env.obs, obs_radius=env.obs_radius)
    assert (
        env.observation_space(params).shape
        == er_env.observation_space(er_params).shape
    )


def test_bootstrap_starts_empty(bootstrap_env) -> None:
    env, params = bootstrap_env
    _, state = env.reset_env(random.PRNGKey(0), params)
    assert not np.asarray(state.player_inventory).any()
    assert int(params.max_timesteps) == MAX_TIMESTEPS


# ---------------------------------------------------------------------------
# Completion-only reward
# ---------------------------------------------------------------------------


def test_bootstrap_max_score_is_one() -> None:
    """The env pays 1.0 per episode at most — completion, nothing else."""
    assert MINER_BOOTSTRAP_MAX_SCORE == 1.0


def test_bootstrap_random_rollout_pays_zero(
    bootstrap_env, bootstrap_step
) -> None:
    """A random policy never completes: total reward is exactly 0.

    (That diagnostic bits latch without paying is asserted by the
    oracle test below, which drives the full ladder.)"""
    env, params = bootstrap_env
    key = random.PRNGKey(5)
    _, state = env.reset_env(key, params)

    total = 0.0
    for _ in range(MAX_TIMESTEPS):
        key, ka, ks = random.split(key, 3)
        action = int(random.randint(ka, (), 0, 30))
        _, state, reward, done, _ = bootstrap_step(ks, state, action, params)
        total += float(reward)
        if bool(done):
            break

    assert total == 0.0


def test_bootstrap_timeout_pays_zero(bootstrap_env, bootstrap_step) -> None:
    """An idle episode runs to the step limit and earns exactly 0."""
    env, params = bootstrap_env
    key = random.PRNGKey(7)
    _, state = env.reset_env(key, params)

    total = 0.0
    done = False
    for _ in range(MAX_TIMESTEPS):
        key, ks = random.split(key)
        _, state, reward, done, _ = bootstrap_step(
            ks, state, int(Action.NOOP), params
        )
        total += float(reward)
        if bool(done):
            break

    assert done  # timeout, not completion
    assert total == 0.0


# ---------------------------------------------------------------------------
# Scripted oracle
# ---------------------------------------------------------------------------


def _place_miners_oracle(state) -> int:
    """Place each held miner on the nearest machine-free ore tile."""
    if int(np.asarray(state.player_inventory[0, int(ItemType.MINER)])) == 0:
        return int(Action.NOOP)
    m = np.asarray(state.map)
    machines = np.asarray(state.machine_types)
    px, py = (int(v) for v in state.player_positions[0])
    h, w = m.shape

    # Adjacent machine-free ore tile: face it, then place.
    for d in _DIR_TO_MOVE:
        off = np.asarray(DIRECTIONS[int(d)])
        tx, ty = px + int(off[0]), py + int(off[1])
        if (
            0 <= tx < w
            and 0 <= ty < h
            and int(m[ty, tx]) in _BLOCK_TO_ITEM
            and int(machines[ty, tx]) == 0
        ):
            if int(state.player_directions[0]) == int(d):
                return int(Action.PLACE_MINER)
            return int(_DIR_TO_FACE[d])

    free_ore = np.isin(m, list(_BLOCK_TO_ITEM)) & (machines == 0)
    assert free_ore.any(), "no machine-free ore tile left"
    return _bfs_step_toward(state, free_ore)


def _bootstrap_oracle(state) -> int:
    """Full loop: mine 6 limestone + 6 silicon, craft 6, place on ore."""
    inv = np.asarray(state.player_inventory[0])
    machines = np.asarray(state.machine_types)
    placed = int((machines == int(Machine.MINER)).sum())
    in_hand = int(inv[int(ItemType.MINER)])
    crafted_total = in_hand + placed

    limestone_block = int(BlockType.LIMESTONE)
    silicon_block = int(BlockType.SILICON)
    need_limestone = int(inv[int(ItemType.LIMESTONE)]) < 6 - crafted_total
    need_silicon = int(inv[int(ItemType.SILICON)]) < 6 - crafted_total

    if need_limestone or need_silicon:
        block = limestone_block if need_limestone else silicon_block
        action = _face_or_mine_adjacent(state, {block})
        if action is not None:
            return action
        return _bfs_step_toward(state, np.asarray(state.map) == block)

    if crafted_total < 6:
        return int(Action.CRAFT_MINER)

    if in_hand > 0:
        return _place_miners_oracle(state)

    return int(Action.NOOP)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bootstrap_oracle_pays_one_on_completing_step_only(
    bootstrap_env, bootstrap_step, seed
) -> None:
    """The oracle's mine -> craft -> place run passes through every
    intermediate event (first/sixth mined of each material, craft,
    placements, 1st-5th producing) — all pay 0. The completing step pays
    exactly 1.0 and ends the episode inside the budget, with the full
    diagnostic ladder latched."""
    env, params = bootstrap_env
    key = random.PRNGKey(seed)
    _, state = env.reset_env(key, params)

    steps = 0
    done = False
    for _ in range(MAX_TIMESTEPS):
        action = _bootstrap_oracle(state)
        key, ks = random.split(key)
        _, state, reward, done, _ = bootstrap_step(ks, state, action, params)
        steps += 1
        if done:
            assert float(reward) == 1.0, "completion must pay exactly 1.0"
            break
        assert float(reward) == 0.0, f"intermediate step {steps} paid {reward}"

    assert done and steps < MAX_TIMESTEPS
    unlocked = int(np.asarray(state.achievements_unlocked).sum())
    assert unlocked == NUM_MINER_BOOTSTRAP_ACHIEVEMENTS
    print(f"\nMinerBootstrap-v1 oracle seed {seed}: solved in {steps} steps")
