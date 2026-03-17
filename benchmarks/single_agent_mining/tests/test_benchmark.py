"""Tests for SingleAgentMiningBenchmark: scoring math and end-to-end execution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.core import BenchmarkLevel, LevelResult
from benchmarks.runner import BenchmarkRunner
from benchmarks.single_agent_mining.benchmark import SingleAgentMiningBenchmark
from benchmarks.single_agent_mining.levels import MINING_LEVELS
from benchmarks.single_agent_mining.scoring import (
    RESOURCE_WEIGHTS,
    aggregate_scores,
    score_items,
)


# ---------------------------------------------------------------------------
# Scoring unit tests
# ---------------------------------------------------------------------------


class TestScoreItems:
    """score_items computes the correct weighted sum."""

    def test_zero_items_scores_zero(self) -> None:
        assert score_items({"coal": 0, "iron": 0, "copper": 0}) == pytest.approx(0.0)

    def test_coal_only(self) -> None:
        assert score_items({"coal": 10}) == pytest.approx(10.0)

    def test_iron_only(self) -> None:
        assert score_items({"iron": 5}) == pytest.approx(10.0)

    def test_copper_only(self) -> None:
        assert score_items({"copper": 3}) == pytest.approx(9.0)

    def test_mixed_resources(self) -> None:
        # 4 coal * 1 + 3 iron * 2 + 2 copper * 3 = 4 + 6 + 6 = 16
        assert score_items({"coal": 4, "iron": 3, "copper": 2}) == pytest.approx(16.0)

    def test_unknown_key_ignored(self) -> None:
        assert score_items({"miner": 100, "coal": 1}) == pytest.approx(1.0)

    def test_weight_ordering_copper_gt_iron_gt_coal(self) -> None:
        assert RESOURCE_WEIGHTS["copper"] > RESOURCE_WEIGHTS["iron"] > RESOURCE_WEIGHTS["coal"]


class TestAggregateScores:
    """aggregate_scores computes the mean of per-level scores."""

    def test_single_level(self) -> None:
        assert aggregate_scores([42.0]) == pytest.approx(42.0)

    def test_mean_of_five(self) -> None:
        assert aggregate_scores([10.0, 20.0, 30.0, 40.0, 50.0]) == pytest.approx(30.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_scores([])


# ---------------------------------------------------------------------------
# SingleAgentMiningBenchmark properties
# ---------------------------------------------------------------------------


class TestBenchmarkProperties:
    """Basic properties of the benchmark object."""

    def test_name(self) -> None:
        assert SingleAgentMiningBenchmark().name == "single_agent_mining"

    def test_num_players(self) -> None:
        assert SingleAgentMiningBenchmark().num_players == 1

    def test_levels_returns_five(self) -> None:
        assert len(SingleAgentMiningBenchmark().levels()) == 5

    def test_levels_are_independent_copies(self) -> None:
        b = SingleAgentMiningBenchmark()
        assert b.levels() is not b.levels()

    def test_score_level_matches_score_items(self) -> None:
        bench = SingleAgentMiningBenchmark()
        items = {"coal": 5, "iron": 3, "copper": 1}
        expected = score_items(items)
        assert bench.score_level(MINING_LEVELS[0], items) == pytest.approx(expected)

    def test_score_is_mean_of_level_scores(self) -> None:
        bench = SingleAgentMiningBenchmark()
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        level_results = [
            LevelResult(
                level_name=f"l{i}",
                items_mined={},
                weighted_score=s,
                timesteps_used=200,
                actions=np.zeros(200, dtype=np.int32),
            )
            for i, s in enumerate(scores)
        ]
        assert bench.score(level_results) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# End-to-end: fast version with reduced timesteps
# ---------------------------------------------------------------------------


class _FastSingleAgentMiningBenchmark(SingleAgentMiningBenchmark):
    """Benchmark variant with 10-step episodes for fast test execution."""

    def levels(self) -> list[BenchmarkLevel]:
        return [
            BenchmarkLevel(
                name=bl.name,
                description=bl.description,
                level=bl.level,
                env_params=bl.env_params.replace(max_timesteps=10),
            )
            for bl in super().levels()
        ]


class TestEndToEnd:
    """End-to-end smoke tests using the fast benchmark variant."""

    @pytest.fixture
    def fast_benchmark(self) -> _FastSingleAgentMiningBenchmark:
        return _FastSingleAgentMiningBenchmark()

    @pytest.fixture
    def noop_policy(self):
        return lambda obs: jnp.array(0)

    def test_run_completes_without_error(
        self, fast_benchmark: _FastSingleAgentMiningBenchmark, noop_policy
    ) -> None:
        result = BenchmarkRunner(seed=0).run(fast_benchmark, policies=[noop_policy])
        assert result is not None

    def test_result_has_five_level_results(
        self, fast_benchmark: _FastSingleAgentMiningBenchmark, noop_policy
    ) -> None:
        result = BenchmarkRunner(seed=0).run(fast_benchmark, policies=[noop_policy])
        assert len(result.level_results) == 5

    def test_level_result_names_match_levels(
        self, fast_benchmark: _FastSingleAgentMiningBenchmark, noop_policy
    ) -> None:
        result = BenchmarkRunner(seed=0).run(fast_benchmark, policies=[noop_policy])
        expected_names = [bl.name for bl in fast_benchmark.levels()]
        actual_names = [lr.level_name for lr in result.level_results]
        assert actual_names == expected_names

    def test_aggregate_score_is_mean_of_level_scores(
        self, fast_benchmark: _FastSingleAgentMiningBenchmark, noop_policy
    ) -> None:
        result = BenchmarkRunner(seed=0).run(fast_benchmark, policies=[noop_policy])
        expected = sum(lr.weighted_score for lr in result.level_results) / 5
        assert result.aggregate_score == pytest.approx(expected)

    def test_all_items_mined_non_negative(
        self, fast_benchmark: _FastSingleAgentMiningBenchmark, noop_policy
    ) -> None:
        result = BenchmarkRunner(seed=0).run(fast_benchmark, policies=[noop_policy])
        for lr in result.level_results:
            for v in lr.items_mined.values():
                assert v >= 0

    def test_mine_policy_outscores_noop(
        self, fast_benchmark: _FastSingleAgentMiningBenchmark
    ) -> None:
        """A policy that always mines should collect more coal than one that noops."""
        runner = BenchmarkRunner(seed=0)
        noop_result = runner.run(fast_benchmark, policies=[lambda obs: jnp.array(0)])
        mine_result = runner.run(fast_benchmark, policies=[lambda obs: jnp.array(5)])

        total_noop = sum(sum(lr.items_mined.values()) for lr in noop_result.level_results)
        total_mine = sum(sum(lr.items_mined.values()) for lr in mine_result.level_results)
        assert total_mine >= total_noop

    def test_benchmark_name_in_result(
        self, fast_benchmark: _FastSingleAgentMiningBenchmark, noop_policy
    ) -> None:
        result = BenchmarkRunner(seed=0).run(fast_benchmark, policies=[noop_policy])
        assert result.benchmark_name == "single_agent_mining"
