"""Compatibility tests for single-agent mining levels.

Each test verifies that a level integrates correctly with factoriax:
the level builds a valid state, the map has the right shape and block
contents, the env can step without error, and the terminal condition is
False at episode start.

These tests are the compatibility boundary between the benchmark module
and factoriax — if anything in the engine changes that breaks level
assumptions, these tests catch it immediately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.constants import BlockType, ItemType
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import global_array

from benchmarks.core import BenchmarkLevel
from benchmarks.single_agent_mining.levels import (
    MINING_LEVELS,
    _LEVEL_1,
    _LEVEL_2,
    _LEVEL_3,
    _LEVEL_4,
    _LEVEL_5,
)


# ---------------------------------------------------------------------------
# Parametrised compatibility tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=MINING_LEVELS, ids=lambda bl: bl.name)
def bench_level(request: pytest.FixtureRequest) -> BenchmarkLevel:
    """Yields each of the five mining levels in turn."""
    return request.param


class TestLevelCompatibility:
    """Each mining level must integrate cleanly with factoriax."""

    def test_build_state_succeeds(self, bench_level: BenchmarkLevel) -> None:
        """build_state must not raise for any level / params combination."""
        state = build_state(bench_level.level, bench_level.env_params)
        assert state is not None

    def test_map_shape_matches_params(self, bench_level: BenchmarkLevel) -> None:
        state = build_state(bench_level.level, bench_level.env_params)
        p = bench_level.env_params
        assert state.map.shape == (p.map_height, p.map_width)

    def test_not_terminal_at_start(self, bench_level: BenchmarkLevel) -> None:
        env = FactoriaXEnv()
        state = build_state(bench_level.level, bench_level.env_params)
        assert not bool(env.is_terminal(state, bench_level.env_params))

    def test_single_step_does_not_crash(self, bench_level: BenchmarkLevel) -> None:
        env = FactoriaXEnv()
        state = build_state(bench_level.level, bench_level.env_params)
        key = jax.random.PRNGKey(0)
        obs, new_state, reward, done, info = env.step_env(
            key, state, jnp.array(0), bench_level.env_params
        )
        assert obs.ndim == 1
        assert obs.dtype == jnp.float32

    def test_observation_shape_matches_env(self, bench_level: BenchmarkLevel) -> None:
        env = FactoriaXEnv()
        state = build_state(bench_level.level, bench_level.env_params)
        expected_dim = env.observation_space(bench_level.env_params).shape[0]
        obs = global_array(state, bench_level.env_params, 0)
        assert obs.shape == (expected_dim,)

    def test_num_players_is_one(self, bench_level: BenchmarkLevel) -> None:
        assert bench_level.env_params.num_players == 1

    def test_max_timesteps_is_200(self, bench_level: BenchmarkLevel) -> None:
        assert bench_level.env_params.max_timesteps == 200

    def test_items_mined_zero_at_start(self, bench_level: BenchmarkLevel) -> None:
        state = build_state(bench_level.level, bench_level.env_params)
        assert jnp.all(state.items_mined == 0)

    def test_level_and_params_dimensions_consistent(self, bench_level: BenchmarkLevel) -> None:
        lvl = bench_level.level
        p = bench_level.env_params
        assert lvl.map_width == p.map_width
        assert lvl.map_height == p.map_height


# ---------------------------------------------------------------------------
# Per-level block content verification
# ---------------------------------------------------------------------------


class TestLevel1Content:
    """Level 1 has coal adjacent to spawn."""

    def test_has_coal_tiles(self) -> None:
        m = np.array(_LEVEL_1.level.block_map)
        assert (m == int(BlockType.COAL)).any()

    def test_no_iron_or_copper(self) -> None:
        m = np.array(_LEVEL_1.level.block_map)
        assert not (m == int(BlockType.IRON)).any()
        assert not (m == int(BlockType.COPPER)).any()

    def test_coal_near_center(self) -> None:
        """Coal patch must exist within 4 tiles of map centre."""
        m = np.array(_LEVEL_1.level.block_map)
        cy, cx = m.shape[0] // 2, m.shape[1] // 2
        coal_rows, coal_cols = np.where(m == int(BlockType.COAL))
        min_dist = int(np.min(np.abs(coal_rows - cy) + np.abs(coal_cols - cx)))
        assert min_dist <= 4


class TestLevel2Content:
    """Level 2 has both coal and iron."""

    def test_has_coal(self) -> None:
        m = np.array(_LEVEL_2.level.block_map)
        assert (m == int(BlockType.COAL)).any()

    def test_has_iron(self) -> None:
        m = np.array(_LEVEL_2.level.block_map)
        assert (m == int(BlockType.IRON)).any()

    def test_no_copper(self) -> None:
        m = np.array(_LEVEL_2.level.block_map)
        assert not (m == int(BlockType.COPPER)).any()


class TestLevel3Content:
    """Level 3 has all three ore types."""

    def test_has_coal(self) -> None:
        m = np.array(_LEVEL_3.level.block_map)
        assert (m == int(BlockType.COAL)).any()

    def test_has_iron(self) -> None:
        m = np.array(_LEVEL_3.level.block_map)
        assert (m == int(BlockType.IRON)).any()

    def test_has_copper(self) -> None:
        m = np.array(_LEVEL_3.level.block_map)
        assert (m == int(BlockType.COPPER)).any()


class TestLevel4Content:
    """Level 4 has iron close and copper far."""

    def test_has_iron(self) -> None:
        m = np.array(_LEVEL_4.level.block_map)
        assert (m == int(BlockType.IRON)).any()

    def test_has_copper(self) -> None:
        m = np.array(_LEVEL_4.level.block_map)
        assert (m == int(BlockType.COPPER)).any()

    def test_iron_closer_to_center_than_copper(self) -> None:
        m = np.array(_LEVEL_4.level.block_map)
        cy, cx = m.shape[0] // 2, m.shape[1] // 2

        iron_rows, iron_cols = np.where(m == int(BlockType.IRON))
        copper_rows, copper_cols = np.where(m == int(BlockType.COPPER))

        iron_min_dist = int(np.min(np.abs(iron_rows - cy) + np.abs(iron_cols - cx)))
        copper_min_dist = int(np.min(np.abs(copper_rows - cy) + np.abs(copper_cols - cx)))

        assert iron_min_dist < copper_min_dist


class TestLevel5Content:
    """Level 5 has iron and copper in opposite corners, no coal."""

    def test_has_iron(self) -> None:
        m = np.array(_LEVEL_5.level.block_map)
        assert (m == int(BlockType.IRON)).any()

    def test_has_copper(self) -> None:
        m = np.array(_LEVEL_5.level.block_map)
        assert (m == int(BlockType.COPPER)).any()

    def test_no_coal(self) -> None:
        m = np.array(_LEVEL_5.level.block_map)
        assert not (m == int(BlockType.COAL)).any()

    def test_map_is_22x22(self) -> None:
        assert _LEVEL_5.level.map_width == 22
        assert _LEVEL_5.level.map_height == 22
