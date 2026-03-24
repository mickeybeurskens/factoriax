"""Tests for the BalancedGatheringBenchmark class and scoring."""

from __future__ import annotations

import numpy as np

from benchmarks.balanced_gathering import BalancedGatheringBenchmark
from benchmarks.balanced_gathering.scoring import (
    aggregate_scores,
    constraint_summary,
    score_items,
)
from benchmarks.core import Benchmark, LevelResult


class TestBenchmarkProtocol:
    """Verify that BalancedGatheringBenchmark satisfies the Benchmark protocol."""

    def test_is_benchmark(self) -> None:
        bench = BalancedGatheringBenchmark()
        assert isinstance(bench, Benchmark)

    def test_name(self) -> None:
        bench = BalancedGatheringBenchmark()
        assert bench.name == "balanced_gathering"

    def test_num_players(self) -> None:
        bench = BalancedGatheringBenchmark()
        assert bench.num_players == 1

    def test_levels_returns_five(self) -> None:
        bench = BalancedGatheringBenchmark()
        assert len(bench.levels()) == 5

    def test_score_level(self) -> None:
        bench = BalancedGatheringBenchmark()
        bl = bench.levels()[0]
        score = bench.score_level(bl, {"coal": 3, "iron": 2, "copper": 1})
        assert score == 6.0

    def test_aggregate_score(self) -> None:
        bench = BalancedGatheringBenchmark()
        results = [
            LevelResult(
                level_name=f"l{i}",
                items_mined={"coal": 1, "iron": 1, "copper": 1},
                weighted_score=3.0,
                timesteps_used=100,
                actions=np.zeros(100, dtype=np.int32),
            )
            for i in range(5)
        ]
        assert bench.score(results) == 3.0


class TestScoring:
    """Tests for scoring functions."""

    def test_score_items_all_types(self) -> None:
        assert score_items({"coal": 5, "iron": 3, "copper": 2}) == 10.0

    def test_score_items_unknown_ignored(self) -> None:
        assert score_items({"coal": 5, "diamond": 100}) == 5.0

    def test_aggregate_scores_mean(self) -> None:
        assert aggregate_scores([2.0, 4.0, 6.0]) == 4.0

    def test_aggregate_scores_empty_raises(self) -> None:
        try:
            aggregate_scores([])
            assert False, "Should have raised"
        except ValueError:
            pass


class TestConstraintSummary:
    """Tests for constraint_summary."""

    def test_all_zero(self) -> None:
        costs = np.zeros((100, 3))
        summary = constraint_summary(costs)
        assert summary["total_cost"] == 0.0
        assert summary["steps_violated"] == 0.0
        assert summary["fraction_violated"] == 0.0

    def test_some_violations(self) -> None:
        costs = np.zeros((100, 3))
        costs[10, 0] = 1.5
        costs[20, 1] = 2.0
        costs[20, 2] = 0.5
        summary = constraint_summary(costs)
        assert summary["total_cost"] == 4.0
        assert summary["steps_violated"] == 2.0
        assert summary["max_step_cost"] == 2.5  # step 20: 2.0 + 0.5
        assert summary["fraction_violated"] == 2.0 / 100.0

    def test_all_violated(self) -> None:
        costs = np.ones((50, 2))
        summary = constraint_summary(costs)
        assert summary["steps_violated"] == 50.0
        assert summary["fraction_violated"] == 1.0
