"""Compatibility tests for single-agent mining levels.

Divided into two layers:

  TestLevelContent  — pure numpy checks on block maps. No JAX, no env
                      interaction. Runs in milliseconds.

  TestLevelJax      — JAX state construction and terminal checks. These use
                      jnp.array operations but no step_env, so there is no
                      JIT compilation overhead. Parametrised over all levels.

  test_env_integration — a single non-parametrised test that calls step_env
                         once (level 1 as representative). This is the only
                         test that triggers JIT compilation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.core import BenchmarkLevel
from benchmarks.single_agent_mining.levels import (
    _LEVEL_1,
    _LEVEL_2,
    _LEVEL_3,
    _LEVEL_4,
    _LEVEL_5,
    MINING_LEVELS,
)
from factoriax.constants import BlockType
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import global_array

# ---------------------------------------------------------------------------
# Content tests — pure numpy, all five levels
# ---------------------------------------------------------------------------


class TestLevelContent:
    """Block map contents are correct for each level. No JAX."""

    # Level 1: coal only, near spawn
    def test_l1_has_coal(self) -> None:
        assert (np.array(_LEVEL_1.level.block_map) == int(BlockType.COAL)).any()

    def test_l1_no_iron_or_copper(self) -> None:
        m = np.array(_LEVEL_1.level.block_map)
        assert not (m == int(BlockType.IRON)).any()
        assert not (m == int(BlockType.COPPER)).any()

    def test_l1_coal_near_center(self) -> None:
        m = np.array(_LEVEL_1.level.block_map)
        cy, cx = m.shape[0] // 2, m.shape[1] // 2
        rows, cols = np.where(m == int(BlockType.COAL))
        assert int(np.min(np.abs(rows - cy) + np.abs(cols - cx))) <= 4

    # Level 2: coal + iron, no copper
    def test_l2_has_coal_and_iron(self) -> None:
        m = np.array(_LEVEL_2.level.block_map)
        assert (m == int(BlockType.COAL)).any()
        assert (m == int(BlockType.IRON)).any()

    def test_l2_no_copper(self) -> None:
        assert not (np.array(_LEVEL_2.level.block_map) == int(BlockType.COPPER)).any()

    # Level 3: all three ore types
    def test_l3_has_all_three_ores(self) -> None:
        m = np.array(_LEVEL_3.level.block_map)
        assert (m == int(BlockType.COAL)).any()
        assert (m == int(BlockType.IRON)).any()
        assert (m == int(BlockType.COPPER)).any()

    # Level 4: iron closer to centre than copper
    def test_l4_iron_closer_than_copper(self) -> None:
        m = np.array(_LEVEL_4.level.block_map)
        cy, cx = m.shape[0] // 2, m.shape[1] // 2
        iron_r, iron_c = np.where(m == int(BlockType.IRON))
        cop_r, cop_c = np.where(m == int(BlockType.COPPER))
        iron_dist = int(np.min(np.abs(iron_r - cy) + np.abs(iron_c - cx)))
        cop_dist = int(np.min(np.abs(cop_r - cy) + np.abs(cop_c - cx)))
        assert iron_dist < cop_dist

    # Level 5: iron + copper in opposite corners, no coal, 22x22
    def test_l5_has_iron_and_copper_no_coal(self) -> None:
        m = np.array(_LEVEL_5.level.block_map)
        assert (m == int(BlockType.IRON)).any()
        assert (m == int(BlockType.COPPER)).any()
        assert not (m == int(BlockType.COAL)).any()

    def test_l5_is_22x22(self) -> None:
        assert _LEVEL_5.level.map_width == 22
        assert _LEVEL_5.level.map_height == 22


# ---------------------------------------------------------------------------
# JAX state tests — all five levels, no step_env (no JIT compilation)
# ---------------------------------------------------------------------------


@pytest.fixture(params=MINING_LEVELS, ids=lambda bl: bl.name)
def bench_level(request: pytest.FixtureRequest) -> BenchmarkLevel:
    return request.param


class TestLevelJax:
    """build_state and derived checks for every level. No step_env."""

    def test_build_state_succeeds(self, bench_level: BenchmarkLevel) -> None:
        assert build_state(bench_level.level, bench_level.env_params) is not None

    def test_map_shape_matches_params(self, bench_level: BenchmarkLevel) -> None:
        state = build_state(bench_level.level, bench_level.env_params)
        p = bench_level.env_params
        assert state.map.shape == (p.map_height, p.map_width)

    def test_items_mined_zero_at_start(self, bench_level: BenchmarkLevel) -> None:
        state = build_state(bench_level.level, bench_level.env_params)
        assert jnp.all(state.items_mined == 0)

    def test_not_terminal_at_start(self, bench_level: BenchmarkLevel) -> None:
        env = FactoriaXEnv()
        state = build_state(bench_level.level, bench_level.env_params)
        assert not bool(env.is_terminal(state, bench_level.env_params))

    def test_num_players_is_one(self, bench_level: BenchmarkLevel) -> None:
        assert bench_level.env_params.num_players == 1

    def test_max_timesteps_is_200(self, bench_level: BenchmarkLevel) -> None:
        assert bench_level.env_params.max_timesteps == 200

    def test_dimensions_consistent(self, bench_level: BenchmarkLevel) -> None:
        lvl, p = bench_level.level, bench_level.env_params
        assert lvl.map_width == p.map_width
        assert lvl.map_height == p.map_height

    def test_observation_shape(self, bench_level: BenchmarkLevel) -> None:
        env = FactoriaXEnv()
        state = build_state(bench_level.level, bench_level.env_params)
        expected = env.observation_space(bench_level.env_params).shape[0]
        obs = global_array(state, bench_level.env_params, 0)
        assert obs.shape == (expected,)


# ---------------------------------------------------------------------------
# Single JAX step test — level 1 only, covers the JIT compilation path
# ---------------------------------------------------------------------------


def test_env_integration() -> None:
    """step_env runs without error and returns a valid float32 observation."""
    env = FactoriaXEnv()
    bl = _LEVEL_1
    state = build_state(bl.level, bl.env_params)
    _, new_state, _, _, _ = env.step_env(
        jax.random.PRNGKey(0), state, jnp.array(0), bl.env_params
    )
    obs = global_array(new_state, bl.env_params, 0)
    assert obs.ndim == 1
    assert obs.dtype == jnp.float32
