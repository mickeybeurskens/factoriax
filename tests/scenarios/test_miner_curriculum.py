"""Tests for the miner-curriculum scenarios.

Scenario-level contracts for the four-stage curriculum that decomposes
EasyRocket-v1's bootstrap phase (see the module docstring of
``factoriax.engine.envs.miner_curriculum``). Covered per scenario:
registration, the transfer invariant (obs shape identical across the
stages and to EasyRocket-v1 built with the same obs variant),
start-state contract, reward accounting against the
latched achievement bits, anti-hack regressions (cycling past the
high-water mark earns nothing), and a scripted-oracle solvability
check within the 300-step budget.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import random

from factoriax.engine.constants import (
    Action,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.envs.easy_rocket import easy_rocket
from factoriax.engine.envs.miner_curriculum import (
    CRAFT_MINERS_MAX_SCORE,
    MINE_ORES_MAX_SCORE,
    MINER_BOOTSTRAP_MAX_SCORE,
    PLACE_MINERS_MAX_SCORE,
    craft_miners,
    mine_ores,
    miner_bootstrap,
    place_miners,
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


def _bfs_step_toward(state, wanted: np.ndarray) -> int:
    """First move of a shortest walkable path to a tile adjacent to
    ``wanted``.

    Machines (all solid on this map — no belts) block movement, which is
    what defeats the greedy walker once placed miners appear: BFS routes
    around them.

    Parameters
    ----------
    wanted :
        Boolean ``(h, w)`` mask of target tiles (stood *next to*, not on).
    """
    from collections import deque

    m = np.asarray(state.map)
    machines = np.asarray(state.machine_types)
    h, w = m.shape
    walkable = (m != int(BlockType.WATER)) & (machines == 0)
    px, py = (int(v) for v in state.player_positions[0])

    goal = np.zeros_like(wanted)
    for d in _DIR_TO_MOVE:
        off = np.asarray(DIRECTIONS[int(d)])
        shifted = np.roll(wanted, (int(off[1]), int(off[0])), axis=(0, 1))
        goal |= shifted
    goal &= walkable

    prev: dict[tuple[int, int], tuple[int, int]] = {}
    seen = {(px, py)}
    queue = deque([(px, py)])
    found = None
    while queue:
        cx, cy = queue.popleft()
        if goal[cy, cx]:
            found = (cx, cy)
            break
        for d in _DIR_TO_MOVE:
            off = np.asarray(DIRECTIONS[int(d)])
            nx, ny = cx + int(off[0]), cy + int(off[1])
            if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx]:
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    prev[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    assert found is not None, "BFS: no reachable tile adjacent to a target"
    cur = found
    while prev.get(cur, (px, py)) != (px, py) and cur in prev:
        cur = prev[cur]
    return _approach(px, py, cur[0], cur[1])


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
    """The transfer invariant: policy weights must load on EasyRocket-v1
    built with the curriculum's obs variant (the scenarios default to
    ``superficial_local``; EasyRocket-v1's own default stays global)."""
    env, params = mine_ores_env
    er_env, er_params = easy_rocket(obs=env.obs, obs_radius=env.obs_radius)
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
    er_env, er_params = easy_rocket(obs=env.obs, obs_radius=env.obs_radius)
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


# ---------------------------------------------------------------------------
# PlaceMiners-v1
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def place_miners_env():
    return place_miners()


@pytest.fixture(scope="module")
def place_miners_step(place_miners_env):
    env, _ = place_miners_env
    return jax.jit(env.step_env)


def test_place_miners_registered() -> None:
    env, params = env_from_name("PlaceMiners-v1")
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape


def test_place_miners_obs_shape_matches_easy_rocket(place_miners_env) -> None:
    env, params = place_miners_env
    er_env, er_params = easy_rocket(obs=env.obs, obs_radius=env.obs_radius)
    assert (
        env.observation_space(params).shape
        == er_env.observation_space(er_params).shape
    )


def test_place_miners_start_inventory(place_miners_env) -> None:
    """Six miners in inventory, nothing else."""
    env, params = place_miners_env
    _, state = env.reset_env(random.PRNGKey(0), params)
    inv = np.asarray(state.player_inventory[0])
    assert int(inv[int(ItemType.MINER)]) == 6
    other = np.delete(inv, [int(ItemType.MINER)])
    assert not other.any()


def test_place_miners_dirt_placement_scores_zero(
    place_miners_env, place_miners_step
) -> None:
    """A miner on dirt never produces, so it earns nothing."""
    env, params = place_miners_env
    key = random.PRNGKey(1)
    _, state = env.reset_env(key, params)

    # Spawn area and its ring are guaranteed dirt: place straight up.
    total = 0.0
    actions = [int(Action.FACE_UP), int(Action.PLACE_MINER)] + [
        int(Action.NOOP)
    ] * 20
    for action in actions:
        key, ks = random.split(key)
        _, state, reward, _, _ = place_miners_step(ks, state, action, params)
        total += float(reward)

    assert total == 0.0
    assert int(np.asarray(state.player_inventory[0, int(ItemType.MINER)])) == 5


def test_place_miners_cycling_past_high_water_earns_zero(
    place_miners_env, place_miners_step
) -> None:
    """Anti-hack: pickup/re-place of a producing miner re-earns nothing."""
    env, params = place_miners_env
    key = random.PRNGKey(2)
    _, state = env.reset_env(key, params)

    def run(action):
        nonlocal key, state
        key, ks = random.split(key)
        _, state, reward, done, _ = place_miners_step(
            ks, state, action, params
        )
        return float(reward), bool(done)

    # Walk to the nearest ore tile and place one miner on it.
    total = 0.0
    for _ in range(60):
        action = _face_or_mine_adjacent(state, set(_BLOCK_TO_ITEM))
        if action == int(Action.MINE):
            action = int(Action.PLACE_MINER)
        if action is None:
            px, py = (int(v) for v in state.player_positions[0])
            target = _nearest(state, set(_BLOCK_TO_ITEM))
            action = _approach(px, py, *target)
        reward, _ = run(action)
        total += reward
        if int(np.asarray(state.player_inventory[0, int(ItemType.MINER)])) < 6:
            break
    # Let it produce: the >=1-producing bit latches.
    for _ in range(3):
        reward, _ = run(int(Action.NOOP))
        total += reward
    assert total == 1.0

    # Pick it up (still facing it) and re-place; let it produce again.
    extra = 0.0
    for action in [int(Action.PICKUP), int(Action.PLACE_MINER)] + [
        int(Action.NOOP)
    ] * 5:
        reward, _ = run(action)
        extra += reward

    assert extra == 0.0


def _place_miners_oracle(state) -> int:
    """Place each miner on the nearest machine-free ore tile."""
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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_place_miners_oracle_reaches_max_and_terminates_early(
    place_miners_env, place_miners_step, seed
) -> None:
    """Oracle places all six miners on ore; episode ends before budget."""
    env, params = place_miners_env
    key = random.PRNGKey(seed)
    _, state = env.reset_env(key, params)

    total = 0.0
    steps = 0
    done = False
    for _ in range(MAX_TIMESTEPS):
        action = _place_miners_oracle(state)
        key, ks = random.split(key)
        _, state, reward, done, _ = place_miners_step(ks, state, action, params)
        total += float(reward)
        steps += 1
        if done:
            break

    assert total == PLACE_MINERS_MAX_SCORE, f"oracle stalled at {total}"
    assert done and steps < MAX_TIMESTEPS
    print(f"\nPlaceMiners-v1 oracle seed {seed}: solved in {steps} steps")


# ---------------------------------------------------------------------------
# MinerBootstrap-v1
# ---------------------------------------------------------------------------


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


def test_bootstrap_reward_matches_latched_bits(
    bootstrap_env, bootstrap_step
) -> None:
    """sum(rewards) == latched bit count over a random rollout."""
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

    assert total == int(np.asarray(state.achievements_unlocked).sum())


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
def test_bootstrap_oracle_reaches_max_and_terminates_early(
    bootstrap_env, bootstrap_step, seed
) -> None:
    """Mine -> craft -> place solves within the 300-step budget."""
    env, params = bootstrap_env
    key = random.PRNGKey(seed)
    _, state = env.reset_env(key, params)

    total = 0.0
    steps = 0
    done = False
    for _ in range(MAX_TIMESTEPS):
        action = _bootstrap_oracle(state)
        key, ks = random.split(key)
        _, state, reward, done, _ = bootstrap_step(ks, state, action, params)
        total += float(reward)
        steps += 1
        if done:
            break

    assert total == MINER_BOOTSTRAP_MAX_SCORE, f"oracle stalled at {total}"
    assert done and steps < MAX_TIMESTEPS
    print(f"\nMinerBootstrap-v1 oracle seed {seed}: solved in {steps} steps")


# ---------------------------------------------------------------------------
# Cross-stage transfer invariant
# ---------------------------------------------------------------------------


def test_curriculum_stages_share_obs_variant_and_shape(
    mine_ores_env, craft_miners_env, place_miners_env, bootstrap_env
) -> None:
    """One obs variant, radius, and shape across all four stages."""
    stages = [mine_ores_env, craft_miners_env, place_miners_env, bootstrap_env]
    assert {env.obs for env, _ in stages} == {"superficial_local"}
    assert len({env.obs_radius for env, _ in stages}) == 1
    assert (
        len({env.observation_space(params).shape for env, params in stages})
        == 1
    )
