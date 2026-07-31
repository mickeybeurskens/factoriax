"""Tests for the MinerBootstrap-v1 scenario.

Scenario-level contracts for the sparse miner-bootstrap task (see the
module docstring of ``factoriax.engine.envs.miner_curriculum``).
Covered: registration, the transfer invariant (obs shape identical to
EasyRocket-v1 built with the same obs variant), start-state contract,
completion-only reward accounting (1.0 exactly once, on the completing
step. Every intermediate event pays 0. A timeout pays 0. The 14 ladder
bits still latch as free diagnostics), and a scripted-oracle
solvability check within the 300-step budget.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
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
from factoriax.engine.envs.common import producing_miners
from factoriax.engine.envs.easy_rocket import easy_rocket
from factoriax.engine.envs.miner_curriculum import (
    MINER_BOOTSTRAP_MAX_SCORE,
    NUM_MINER_BOOTSTRAP_ACHIEVEMENTS,
    BootstrapStart,
    all_miners_producing,
    miner_bootstrap,
)
from factoriax.engine.placement import place_machine
from factoriax.engine.tables import BLOCK_TO_ITEM_ARRAY, DIRECTIONS
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
        env.observation_space(params).shape == er_env.observation_space(er_params).shape
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
    """The env pays 1.0 per episode at most, for completion and nothing else."""
    assert MINER_BOOTSTRAP_MAX_SCORE == 1.0


def test_bootstrap_random_rollout_pays_zero(bootstrap_env, bootstrap_step) -> None:
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
        _, state, reward, done, _ = bootstrap_step(ks, state, int(Action.NOOP), params)
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
    placements, 1st-5th producing). All of them pay 0. The completing step pays
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


# ---------------------------------------------------------------------------
# Backward-curriculum start states (BootstrapStart / apply_start)
# ---------------------------------------------------------------------------

_N_MINERS = 6


def test_bootstrap_start_classmethods_encode_the_stage_ladder() -> None:
    """A_k / B_k / C_k with k missing miners map onto the three counts."""
    for k in range(1, _N_MINERS + 1):
        assert BootstrapStart.place(k) == BootstrapStart(
            n_placed=_N_MINERS - k, n_miners=k, n_materials=0
        )
        assert BootstrapStart.craft(k) == BootstrapStart(
            n_placed=_N_MINERS - k, n_miners=0, n_materials=k
        )
        assert BootstrapStart.mine(k) == BootstrapStart(
            n_placed=_N_MINERS - k, n_miners=0, n_materials=0
        )


@pytest.mark.parametrize("group", ["place", "craft", "mine"])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_stage_start_state_contract(bootstrap_env, group, k) -> None:
    """Every stage resets to 6-k producing miners, the group's inventory,
    an unfinished episode, and the pristine env's obs shape."""
    pristine_env, params = bootstrap_env
    start = getattr(BootstrapStart, group)(k)
    env, _ = miner_bootstrap(start=start)

    key = random.PRNGKey(20 + k)
    obs, state = env.reset_env(key, params)
    pristine_obs, pristine_state = pristine_env.reset_env(key, params)

    # Transfer invariant: the stage changes nothing the agent sees
    # structurally. The obs shape matches the unmodified env.
    assert obs.shape == pristine_obs.shape

    # 6-k miners pre-placed and producing from step 0.
    producing, _ = producing_miners(state)
    assert int(jnp.sum(producing)) == _N_MINERS - k
    machines = np.asarray(state.machine_types)
    assert int((machines == int(Machine.MINER)).sum()) == _N_MINERS - k

    # Group inventory: A_k holds k miners, B_k holds k of each material,
    # C_k holds nothing. Nothing else is stocked.
    inv = np.asarray(state.player_inventory[0])
    n_miners = k if group == "place" else 0
    n_materials = k if group == "craft" else 0
    assert int(inv[int(ItemType.MINER)]) == n_miners
    assert int(inv[int(ItemType.LIMESTONE)]) == n_materials
    assert int(inv[int(ItemType.SILICON)]) == n_materials
    other = np.delete(
        inv,
        [
            int(ItemType.MINER),
            int(ItemType.LIMESTONE),
            int(ItemType.SILICON),
        ],
    )
    assert not other.any()

    # The episode is not already over and nothing is latched.
    assert not bool(all_miners_producing(state, params))
    assert not np.asarray(state.achievements_unlocked).any()

    # The hook leaves the player where the pristine reset put it.
    np.testing.assert_array_equal(
        np.asarray(state.player_positions),
        np.asarray(pristine_state.player_positions),
    )
    np.testing.assert_array_equal(
        np.asarray(state.player_directions),
        np.asarray(pristine_state.player_directions),
    )


def test_pre_installed_miner_equals_action_path_placement(
    bootstrap_env,
) -> None:
    """The hook's miner is indistinguishable from one placed by the
    place action on the same tile (plus the documented buffer pre-fill)."""
    pristine_env, params = bootstrap_env
    env, _ = miner_bootstrap(start=BootstrapStart.place(5))  # one pre-placed

    key = random.PRNGKey(3)
    _, hooked = env.reset_env(key, params)
    _, pristine = pristine_env.reset_env(key, params)
    np.testing.assert_array_equal(np.asarray(hooked.map), np.asarray(pristine.map))

    machines = np.asarray(hooked.machine_types)
    ys, xs = np.nonzero(machines == int(Machine.MINER))
    assert len(ys) == 1
    ty, tx = int(ys[0]), int(xs[0])

    # Action path on the pristine state: stand above the tile, face
    # down, place, then pre-fill the buffer with one unit of the ore.
    s = pristine.replace(
        player_inventory=pristine.player_inventory.at[0, int(ItemType.MINER)].set(1),
        player_positions=pristine.player_positions.at[0].set(
            jnp.asarray([tx, ty - 1], dtype=pristine.player_positions.dtype)
        ),
        player_directions=pristine.player_directions.at[0].set(
            jnp.asarray(int(Direction.DOWN), dtype=pristine.player_directions.dtype)
        ),
    )
    s = place_machine(s, params, 0, int(ItemType.MINER))
    idx = int(np.asarray(s.tile_entity[ty, tx]))
    assert idx >= 0, "action-path placement failed"
    ore_item = int(BLOCK_TO_ITEM_ARRAY[int(np.asarray(s.map[ty, tx]))])
    s = s.replace(
        ent_buf_type=s.ent_buf_type.at[idx].set(jnp.int8(ore_item)),
        ent_buf_count=s.ent_buf_count.at[idx].set(jnp.int16(1)),
    )

    for field in (
        "machine_types",
        "tile_entity",
        "ent_y",
        "ent_x",
        "ent_type",
        "ent_direction",
        "ent_health",
        "ent_buf_type",
        "ent_buf_count",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(s, field)),
            np.asarray(getattr(hooked, field)),
            err_msg=field,
        )


def test_pre_placed_patches_vary_with_the_reset_key() -> None:
    """Which patches host the pre-placed miners is drawn per episode, so
    the agent cannot memorize the free patch's ore type."""
    env, params = miner_bootstrap(start=BootstrapStart.place(1))

    free_ores = set()
    for seed in range(8):
        _, state = env.reset_env(random.PRNGKey(seed), params)
        m = np.asarray(state.map)
        machines = np.asarray(state.machine_types)
        occupied = np.unique(m[machines == int(Machine.MINER)])
        patch_blocks = np.unique(m[m != int(BlockType.DIRT)])
        free_ores.update(int(b) for b in patch_blocks if b not in occupied)
    assert len(free_ores) > 1


@pytest.mark.parametrize(
    "stage",
    ["place_1", "craft_1", "mine_1", "mine_6"],
)
def test_stage_oracle_completes_within_budget(stage) -> None:
    """The scripted oracle earns the completion reward from the easiest
    stage of each group and from the pristine task, 3 seeds each."""
    group, k = stage.rsplit("_", 1)
    env, params = miner_bootstrap(start=getattr(BootstrapStart, group)(int(k)))
    step = jax.jit(env.step_env)

    for seed in range(3):
        key = random.PRNGKey(seed)
        _, state = env.reset_env(key, params)

        steps = 0
        done = False
        for _ in range(MAX_TIMESTEPS):
            action = _bootstrap_oracle(state)
            key, ks = random.split(key)
            _, state, reward, done, _ = step(ks, state, action, params)
            steps += 1
            if done:
                assert float(reward) == 1.0
                break
            assert float(reward) == 0.0

        assert done and steps < MAX_TIMESTEPS, (
            f"{stage} seed {seed}: oracle did not finish (steps={steps})"
        )
        print(f"\n{stage} oracle seed {seed}: solved in {steps} steps")
