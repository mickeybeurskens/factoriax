"""Tests for SingleAgentMiningBenchmark: scoring math and end-to-end execution.

Scoring tests are pure Python — instant. End-to-end tests share a single
session-scoped BenchmarkResult so the runner is built and the benchmark is
executed exactly once. The fast benchmark uses five copies of the same 10×10
level so JIT compiles once for that shape.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from factoriax.benchmarks.core import BenchmarkLevel, BenchmarkResult, LevelResult
from factoriax.benchmarks.runner import BenchmarkRunner
from factoriax.benchmarks.single_agent_mining.benchmark import (
    SingleAgentMiningBenchmark,
)
from factoriax.benchmarks.single_agent_mining.levels import MINING_LEVELS
from factoriax.benchmarks.single_agent_mining.scoring import (
    RESOURCE_WEIGHTS,
    aggregate_scores,
    score_items,
)
from factoriax.constants import BlockType, NUM_ITEM_TYPES, ItemType
from factoriax.levels import LevelBuilder, generate_state
from factoriax.rewards import sparse_mining_reward
from factoriax.state import EnvParams

# ---------------------------------------------------------------------------
# Scoring unit tests — pure Python, no JAX
# ---------------------------------------------------------------------------


class TestScoreItems:
    def test_zero_items(self) -> None:
        assert score_items({"coal": 0, "iron": 0, "copper": 0}) == pytest.approx(0.0)

    def test_coal_only(self) -> None:
        assert score_items({"coal": 10}) == pytest.approx(10.0)

    def test_iron_only(self) -> None:
        assert score_items({"iron": 5}) == pytest.approx(5.0)

    def test_copper_only(self) -> None:
        assert score_items({"copper": 3}) == pytest.approx(3.0)

    def test_mixed(self) -> None:
        # 4 + 3 + 2 = 9
        assert score_items({"coal": 4, "iron": 3, "copper": 2}) == pytest.approx(9.0)

    def test_unknown_key_ignored(self) -> None:
        assert score_items({"miner": 100, "coal": 1}) == pytest.approx(1.0)

    def test_equal_weights(self) -> None:
        assert RESOURCE_WEIGHTS["coal"] == RESOURCE_WEIGHTS["iron"] == RESOURCE_WEIGHTS["copper"]


class TestAggregateScores:
    def test_single_level(self) -> None:
        assert aggregate_scores([42.0]) == pytest.approx(42.0)

    def test_mean_of_five(self) -> None:
        assert aggregate_scores([10.0, 20.0, 30.0, 40.0, 50.0]) == pytest.approx(30.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_scores([])


# ---------------------------------------------------------------------------
# Benchmark object properties — no JAX
# ---------------------------------------------------------------------------


class TestBenchmarkProperties:
    def test_name(self) -> None:
        assert SingleAgentMiningBenchmark().name == "single_agent_mining"

    def test_num_players(self) -> None:
        assert SingleAgentMiningBenchmark().num_players == 1

    def test_levels_count(self) -> None:
        assert len(SingleAgentMiningBenchmark().levels()) == 5

    def test_levels_are_independent_copies(self) -> None:
        b = SingleAgentMiningBenchmark()
        assert b.levels() is not b.levels()

    def test_score_level_matches_score_items(self) -> None:
        items = {"coal": 5, "iron": 3, "copper": 1}
        bench = SingleAgentMiningBenchmark()
        assert bench.score_level(MINING_LEVELS[0], items) == pytest.approx(
            score_items(items)
        )

    def test_reward_fn_is_sparse_mining_reward(self) -> None:
        """reward_fn property must return sparse_mining_reward."""
        assert SingleAgentMiningBenchmark().reward_fn is sparse_mining_reward

    def test_reward_fn_returns_zero_with_no_mining(self, state_factory) -> None:
        """reward_fn returns 0.0 when no ore was extracted."""
        state = state_factory(world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32))
        params = EnvParams()
        reward = SingleAgentMiningBenchmark().reward_fn(state, state, params)
        assert float(reward) == 0.0

    def test_reward_fn_counts_ore_mined(self, state_factory) -> None:
        """reward_fn returns 1.0 per ore item extracted."""
        world_map = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
        prev = state_factory(world_map=world_map)
        items = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32).at[ItemType.COAL].set(2)
        new = state_factory(world_map=world_map, items_mined=items)
        params = EnvParams()
        reward = SingleAgentMiningBenchmark().reward_fn(prev, new, params)
        assert float(reward) == pytest.approx(2.0)

    def test_score_is_mean(self) -> None:
        level_results = [
            LevelResult(f"l{i}", {}, s, 200, np.zeros(200, dtype=np.int32))
            for i, s in enumerate([10.0, 20.0, 30.0, 40.0, 50.0])
        ]
        assert SingleAgentMiningBenchmark().score(level_results) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# End-to-end — single shared run, many assertions
#
# _FastBenchmark uses five copies of the same 10x10 level so JIT compiles
# once. The session-scoped result fixture runs the benchmark exactly once
# for the entire test session.
# ---------------------------------------------------------------------------


def _make_fast_level() -> BenchmarkLevel:
    """A 10×10 coal level with 5-step episodes for minimal JAX overhead."""
    level = LevelBuilder(10, 10).fill_rect(2, 2, 3, 3, BlockType.COAL).build("fast")
    params = EnvParams(map_width=10, map_height=10, num_players=1, max_timesteps=5)
    return BenchmarkLevel(
        name="fast_level",
        description="Fast test level.",
        level=level,
        env_params=params,
    )


class _FastBenchmark(SingleAgentMiningBenchmark):
    """Five copies of the same 10×10 level — single JIT compilation."""

    def levels(self) -> list[BenchmarkLevel]:
        bl = _make_fast_level()
        return [
            BenchmarkLevel(
                name=f"fast_{i}",
                description=bl.description,
                level=bl.level,
                env_params=bl.env_params,
            )
            for i in range(5)
        ]


@pytest.fixture(scope="session")
def fast_result() -> BenchmarkResult:
    """Run the fast benchmark once for the entire session."""
    return BenchmarkRunner(seed=0).run(
        _FastBenchmark(), policies=[lambda obs: jnp.array(0)]
    )


class TestEndToEnd:
    """Assertions against the single shared session result."""

    def test_five_level_results(self, fast_result: BenchmarkResult) -> None:
        assert len(fast_result.level_results) == 5

    def test_benchmark_name(self, fast_result: BenchmarkResult) -> None:
        assert fast_result.benchmark_name == "single_agent_mining"

    def test_aggregate_is_mean_of_level_scores(
        self, fast_result: BenchmarkResult
    ) -> None:
        expected = sum(lr.weighted_score for lr in fast_result.level_results) / 5
        assert fast_result.aggregate_score == pytest.approx(expected)

    def test_items_mined_non_negative(self, fast_result: BenchmarkResult) -> None:
        for lr in fast_result.level_results:
            assert all(v >= 0 for v in lr.items_mined.values())

    def test_items_mined_keys_present(self, fast_result: BenchmarkResult) -> None:
        for lr in fast_result.level_results:
            assert set(lr.items_mined.keys()) == {"coal", "iron", "copper"}

    def test_timesteps_at_most_max(self, fast_result: BenchmarkResult) -> None:
        for lr in fast_result.level_results:
            assert lr.timesteps_used <= 5

    def test_actions_length_matches_timesteps(
        self, fast_result: BenchmarkResult
    ) -> None:
        for lr in fast_result.level_results:
            assert lr.actions.shape == (lr.timesteps_used,)


def test_mine_policy_outscores_noop() -> None:
    """Always-mine beats always-noop on a level with adjacent coal."""
    bench = _FastBenchmark()
    runner = BenchmarkRunner(seed=0)
    noop = runner.run(bench, policies=[lambda obs: jnp.array(0)])
    mine = runner.run(bench, policies=[lambda obs: jnp.array(5)])

    def total(r):
        return sum(sum(lr.items_mined.values()) for lr in r.level_results)

    assert total(mine) >= total(noop)
