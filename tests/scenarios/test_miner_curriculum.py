"""Tests for the miner-curriculum scenarios.

Scenario-level contracts for the four-stage curriculum that decomposes
EasyRocket-v1's bootstrap phase (see the module docstring of
``factoriax.engine.envs.miner_curriculum``). Covered per scenario:
registration, the transfer invariant (obs shape identical to
EasyRocket-v1), start-state contract, reward accounting against the
latched achievement bits, anti-hack regressions (cycling past the
high-water mark earns nothing), and a scripted-oracle solvability
check within the 300-step budget.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import random

from factoriax.engine.constants import Action, BlockType, Direction, ItemType
from factoriax.engine.envs.easy_rocket import easy_rocket
from factoriax.engine.envs.miner_curriculum import (
    CRAFT_MINERS_MAX_SCORE,
    MINE_ORES_MAX_SCORE,
    craft_miners,
    mine_ores,
)
from factoriax.engine.tables import DIRECTIONS
from factoriax.make import env_from_name

MAX_TIMESTEPS = 300

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

#: Ore block on the map -> ore item it yields when mined.
_BLOCK_TO_ITEM = {
    int(BlockType.IRON): int(ItemType.IRON_ORE),
    int(BlockType.COPPER): int(ItemType.COPPER_ORE),
    int(BlockType.TIN): int(ItemType.TIN_ORE),
    int(BlockType.SILICON): int(ItemType.SILICON),
    int(BlockType.COAL): int(ItemType.COAL),
    int(BlockType.LIMESTONE): int(ItemType.LIMESTONE),
}


def _approach(px: int, py: int, tx: int, ty: int) -> int:
    """Greedy move action toward ``(tx, ty)``, longer axis first."""
    dx, dy = tx - px, ty - py
    if abs(dx) >= abs(dy):
        return int(_DIR_TO_MOVE[Direction.RIGHT if dx > 0 else Direction.LEFT])
    return int(_DIR_TO_MOVE[Direction.DOWN if dy > 0 else Direction.UP])


def _face_or_mine_adjacent(state, wanted_blocks: set[int]) -> int | None:
    """MINE if facing a wanted block; FACE it if adjacent; else None."""
    m = np.asarray(state.map)
    h, w = m.shape
    px, py = (int(v) for v in state.player_positions[0])
    for d in _DIR_TO_MOVE:
        off = np.asarray(DIRECTIONS[int(d)])
        tx, ty = px + int(off[0]), py + int(off[1])
        if 0 <= tx < w and 0 <= ty < h and int(m[ty, tx]) in wanted_blocks:
            if int(state.player_directions[0]) == int(d):
                return int(Action.MINE)
            return int(_DIR_TO_FACE[d])
    return None


def _nearest(state, wanted_blocks: set[int]) -> tuple[int, int] | None:
    """Coordinates of the nearest wanted block, or None."""
    m = np.asarray(state.map)
    px, py = (int(v) for v in state.player_positions[0])
    ys, xs = np.where(np.isin(m, list(wanted_blocks)))
    if len(xs) == 0:
        return None
    dists = np.abs(xs - px) + np.abs(ys - py)
    i = int(np.argmin(dists))
    return int(xs[i]), int(ys[i])


# ---------------------------------------------------------------------------
# MineOres-v1
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mine_ores_env():
    return mine_ores()


@pytest.fixture(scope="module")
def mine_ores_step(mine_ores_env):
    env, _ = mine_ores_env
    return jax.jit(env.step_env)


def test_mine_ores_registered() -> None:
    env, params = env_from_name("MineOres-v1")
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape


def test_mine_ores_obs_shape_matches_easy_rocket(mine_ores_env) -> None:
    """The transfer invariant: policy weights must load on EasyRocket-v1."""
    env, params = mine_ores_env
    er_env, er_params = easy_rocket()
    assert env.obs == er_env.obs
    assert (
        env.observation_space(params).shape
        == er_env.observation_space(er_params).shape
    )


def test_mine_ores_start_state_contract(mine_ores_env) -> None:
    """Six 2x2 ore patches, empty inventory, 300-step budget."""
    env, params = mine_ores_env
    assert int(params.max_timesteps) == MAX_TIMESTEPS
    for seed in range(3):
        _, state = env.reset_env(random.PRNGKey(seed), params)
        m = np.asarray(state.map)
        for block in _BLOCK_TO_ITEM:
            assert int((m == block).sum()) == 4
        assert not np.asarray(state.player_inventory).any()


def test_mine_ores_reward_matches_latched_bits(
    mine_ores_env, mine_ores_step
) -> None:
    """sum(rewards) == latched bit count == clamped per-ore mining progress."""
    env, params = mine_ores_env
    key = random.PRNGKey(11)
    _, state = env.reset_env(key, params)

    total = 0.0
    dones = []
    for _ in range(MAX_TIMESTEPS):
        key, ka, ks = random.split(key, 3)
        action = int(random.randint(ka, (), 0, 18))
        _, state, reward, done, _ = mine_ores_step(ks, state, action, params)
        total += float(reward)
        dones.append(bool(done))

    mined = np.asarray(state.items_mined)
    expected = sum(min(int(mined[item]), 5) for item in _BLOCK_TO_ITEM.values())
    assert total == expected
    assert total == int(np.asarray(state.achievements_unlocked).sum())
    assert not any(dones[:-1])
    assert dones[-1]


def _mine_ores_oracle(state) -> int:
    """Greedy miner: work on any ore type still short of 5 mined items."""
    mined = np.asarray(state.items_mined)
    wanted = {
        block for block, item in _BLOCK_TO_ITEM.items() if int(mined[item]) < 5
    }
    if not wanted:
        return int(Action.NOOP)
    action = _face_or_mine_adjacent(state, wanted)
    if action is not None:
        return action
    target = _nearest(state, wanted)
    assert target is not None, "wanted ore type missing from the map"
    px, py = (int(v) for v in state.player_positions[0])
    return _approach(px, py, *target)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mine_ores_oracle_reaches_max_score(
    mine_ores_env, mine_ores_step, seed
) -> None:
    """A greedy scripted policy mines 5 of all six ores within the budget."""
    env, params = mine_ores_env
    key = random.PRNGKey(seed)
    _, state = env.reset_env(key, params)

    total = 0.0
    steps = 0
    for _ in range(MAX_TIMESTEPS):
        action = _mine_ores_oracle(state)
        key, ks = random.split(key)
        _, state, reward, _, _ = mine_ores_step(ks, state, action, params)
        total += float(reward)
        steps += 1
        if total >= MINE_ORES_MAX_SCORE:
            break

    assert total == MINE_ORES_MAX_SCORE, f"oracle stalled at {total}"
    print(f"\nMineOres-v1 oracle seed {seed}: solved in {steps} steps")


# ---------------------------------------------------------------------------
# CraftMiners-v1
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def craft_miners_env():
    return craft_miners()


@pytest.fixture(scope="module")
def craft_miners_step(craft_miners_env):
    env, _ = craft_miners_env
    return jax.jit(env.step_env)


def test_craft_miners_registered() -> None:
    env, params = env_from_name("CraftMiners-v1")
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape


def test_craft_miners_obs_shape_matches_easy_rocket(craft_miners_env) -> None:
    env, params = craft_miners_env
    er_env, er_params = easy_rocket()
    assert env.obs == er_env.obs
    assert (
        env.observation_space(params).shape
        == er_env.observation_space(er_params).shape
    )


def test_craft_miners_start_inventory(craft_miners_env) -> None:
    """Exactly the materials for six miners: 6 limestone + 6 silicon."""
    env, params = craft_miners_env
    _, state = env.reset_env(random.PRNGKey(0), params)
    inv = np.asarray(state.player_inventory[0])
    assert int(inv[int(ItemType.LIMESTONE)]) == 6
    assert int(inv[int(ItemType.SILICON)]) == 6
    other = np.delete(inv, [int(ItemType.LIMESTONE), int(ItemType.SILICON)])
    assert not other.any()


def test_craft_miners_six_crafts_reach_max_score(
    craft_miners_env, craft_miners_step
) -> None:
    """Six CRAFT_MINER actions consume all materials and score 6."""
    env, params = craft_miners_env
    key = random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    total = 0.0
    for _ in range(6):
        key, ks = random.split(key)
        _, state, reward, _, _ = craft_miners_step(
            ks, state, int(Action.CRAFT_MINER), params
        )
        total += float(reward)

    assert total == CRAFT_MINERS_MAX_SCORE
    inv = np.asarray(state.player_inventory[0])
    assert int(inv[int(ItemType.MINER)]) == 6
    assert int(inv[int(ItemType.LIMESTONE)]) == 0
    assert int(inv[int(ItemType.SILICON)]) == 0


def test_craft_miners_cycling_past_high_water_earns_zero(
    craft_miners_env, craft_miners_step
) -> None:
    """Anti-hack: place/pickup cycling after max score earns nothing.

    The bits latch on the >=k inventory thresholds; dropping below by
    placing a miner and coming back up via pickup must not re-earn.
    """
    env, params = craft_miners_env
    key = random.PRNGKey(3)
    _, state = env.reset_env(key, params)

    for _ in range(6):
        key, ks = random.split(key)
        _, state, _, _, _ = craft_miners_step(
            ks, state, int(Action.CRAFT_MINER), params
        )
    assert int(np.asarray(state.achievements_unlocked).sum()) == 6

    # Face a dirt neighbour, place a miner (inventory 6 -> 5), pick it
    # back up (5 -> 6), repeatedly. No new reward may appear.
    cycle = [
        int(Action.FACE_UP),
        int(Action.PLACE_MINER),
        int(Action.PICKUP),
        int(Action.PLACE_MINER),
        int(Action.PICKUP),
    ]
    extra = 0.0
    for action in cycle:
        key, ks = random.split(key)
        _, state, reward, _, _ = craft_miners_step(ks, state, action, params)
        extra += float(reward)

    assert extra == 0.0
    assert int(np.asarray(state.player_inventory[0, int(ItemType.MINER)])) == 6
