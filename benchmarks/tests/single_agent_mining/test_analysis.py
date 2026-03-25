"""Tests for single_agent_mining analysis functions.

All tests use synthetic BenchmarkResult objects with known values, so they
run without any environment interaction. Figures are inspected for type and
basic structure; visual correctness is verified manually.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmarks.core import BenchmarkResult, LevelResult
from benchmarks.single_agent_mining.analysis import (
    log_to_wandb,
    plot_action_distribution,
    plot_level_scores,
    plot_resource_breakdown,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    scores: list[float] | None = None,
    items: list[dict[str, int]] | None = None,
) -> BenchmarkResult:
    """Build a synthetic BenchmarkResult with 5 levels."""
    if scores is None:
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    if items is None:
        items = [
            {"coal": 10, "iron": 0, "copper": 0},
            {"coal": 5, "iron": 5, "copper": 0},
            {"coal": 2, "iron": 4, "copper": 4},
            {"coal": 0, "iron": 5, "copper": 5},
            {"coal": 0, "iron": 0, "copper": 10},
        ]
    level_results = [
        LevelResult(
            level_name=f"level_{i + 1}",
            items_mined=items[i],
            weighted_score=scores[i],
            timesteps_used=200,
            actions=np.random.randint(0, 12, size=200, dtype=np.int32),
        )
        for i in range(5)
    ]
    return BenchmarkResult(
        benchmark_name="single_agent_mining",
        level_results=level_results,
        aggregate_score=sum(scores) / len(scores),
    )


@pytest.fixture
def result() -> BenchmarkResult:
    return _make_result()


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_level_scores
# ---------------------------------------------------------------------------


class TestPlotLevelScores:
    """plot_level_scores returns a valid figure."""

    def test_returns_figure(self, result: BenchmarkResult) -> None:
        fig = plot_level_scores(result)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_one_axes(self, result: BenchmarkResult) -> None:
        fig = plot_level_scores(result)
        assert len(fig.axes) == 1

    def test_handles_all_zero_scores(self) -> None:
        r = _make_result(scores=[0.0, 0.0, 0.0, 0.0, 0.0])
        fig = plot_level_scores(r)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_resource_breakdown
# ---------------------------------------------------------------------------


class TestPlotResourceBreakdown:
    """plot_resource_breakdown produces a stacked bar figure."""

    def test_returns_figure(self, result: BenchmarkResult) -> None:
        fig = plot_resource_breakdown(result)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_one_axes(self, result: BenchmarkResult) -> None:
        fig = plot_resource_breakdown(result)
        assert len(fig.axes) == 1

    def test_handles_no_resources_mined(self) -> None:
        r = _make_result(
            items=[{"coal": 0, "iron": 0, "copper": 0}] * 5,
            scores=[0.0] * 5,
        )
        fig = plot_resource_breakdown(r)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_action_distribution
# ---------------------------------------------------------------------------


class TestPlotActionDistribution:
    """plot_action_distribution produces a grouped bar figure."""

    def test_returns_figure(self, result: BenchmarkResult) -> None:
        fig = plot_action_distribution(result)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_one_axes(self, result: BenchmarkResult) -> None:
        fig = plot_action_distribution(result)
        assert len(fig.axes) == 1

    def test_handles_empty_action_array(self) -> None:
        """Levels with zero steps should not cause division by zero."""
        level_results = [
            LevelResult(
                level_name="empty",
                items_mined={"coal": 0, "iron": 0, "copper": 0},
                weighted_score=0.0,
                timesteps_used=0,
                actions=np.array([], dtype=np.int32),
            )
        ]
        r = BenchmarkResult(
            benchmark_name="single_agent_mining",
            level_results=level_results,
            aggregate_score=0.0,
        )
        fig = plot_action_distribution(r)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# log_to_wandb
# ---------------------------------------------------------------------------


class TestLogToWandb:
    """log_to_wandb is a no-op when wandb_run is None."""

    def test_no_op_when_wandb_run_is_none(self, result: BenchmarkResult) -> None:
        """Must not raise or import wandb when wandb_run is None."""
        log_to_wandb(result, wandb_run=None)

    def test_no_op_with_step(self, result: BenchmarkResult) -> None:
        log_to_wandb(result, wandb_run=None, step=1000)
