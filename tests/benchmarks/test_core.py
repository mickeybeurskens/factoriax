"""Tests for benchmarks.core: types and the Benchmark protocol."""

from __future__ import annotations

import numpy as np
import pytest

from factoriax.benchmarks.core import (
    Benchmark,
    BenchmarkLevel,
    BenchmarkResult,
    LevelResult,
)
from factoriax.constants import BlockType
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bench_level(name: str = "test") -> BenchmarkLevel:
    level = LevelBuilder(8, 8).fill_rect(0, 0, 3, 3, BlockType.COAL).build(name)
    params = EnvParams(map_width=8, map_height=8, num_players=1, max_timesteps=10)
    return BenchmarkLevel(
        name=name, description="A test level.", level=level, env_params=params
    )


def _make_level_result(name: str = "test", score: float = 5.0) -> LevelResult:
    return LevelResult(
        level_name=name,
        items_mined={"coal": 5, "iron": 0, "copper": 0},
        weighted_score=score,
        timesteps_used=10,
        actions=np.array([0, 5, 5, 5, 0], dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# BenchmarkLevel
# ---------------------------------------------------------------------------


class TestBenchmarkLevel:
    """BenchmarkLevel stores level and params correctly."""

    def test_fields_accessible(self) -> None:
        bl = _make_bench_level("my_level")
        assert bl.name == "my_level"
        assert bl.description == "A test level."
        assert bl.level.map_width == 8
        assert bl.env_params.num_players == 1

    def test_env_params_max_timesteps(self) -> None:
        bl = _make_bench_level()
        assert bl.env_params.max_timesteps == 10

    def test_blocked_actions_default_none(self) -> None:
        """Default ``blocked_actions`` is ``None`` (fall back to benchmark)."""
        bl = _make_bench_level()
        assert bl.blocked_actions is None

    def test_blocked_actions_accepts_frozenset(self) -> None:
        """Per-level mask can be set at construction time."""
        level = LevelBuilder(8, 8).fill_rect(0, 0, 3, 3, BlockType.COAL).build("masked")
        params = EnvParams(map_width=8, map_height=8, num_players=1, max_timesteps=10)
        bl = BenchmarkLevel(
            name="masked",
            description="Masked level.",
            level=level,
            env_params=params,
            blocked_actions=frozenset({5, 9, 10}),
        )
        assert bl.blocked_actions == frozenset({5, 9, 10})

    def test_blocked_actions_accepts_empty_override(self) -> None:
        """Empty ``frozenset()`` is distinct from ``None`` — overrides class-level."""
        level = LevelBuilder(8, 8).fill_rect(0, 0, 3, 3, BlockType.COAL).build("clear")
        params = EnvParams(map_width=8, map_height=8, num_players=1, max_timesteps=10)
        bl = BenchmarkLevel(
            name="clear",
            description="Explicitly unmasked level.",
            level=level,
            env_params=params,
            blocked_actions=frozenset(),
        )
        assert bl.blocked_actions == frozenset()
        assert bl.blocked_actions is not None


# ---------------------------------------------------------------------------
# LevelResult
# ---------------------------------------------------------------------------


class TestLevelResult:
    """LevelResult stores run outcomes correctly."""

    def test_fields_accessible(self) -> None:
        lr = _make_level_result("lvl1", 7.5)
        assert lr.level_name == "lvl1"
        assert lr.weighted_score == pytest.approx(7.5)
        assert lr.items_mined["coal"] == 5
        assert lr.timesteps_used == 10

    def test_actions_shape(self) -> None:
        lr = _make_level_result()
        assert lr.actions.ndim == 1
        assert lr.actions.dtype == np.int32


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """BenchmarkResult aggregates level results correctly."""

    def test_fields_accessible(self) -> None:
        lr1 = _make_level_result("l1", 10.0)
        lr2 = _make_level_result("l2", 20.0)
        br = BenchmarkResult(
            benchmark_name="test_bench",
            level_results=[lr1, lr2],
            aggregate_score=15.0,
        )
        assert br.benchmark_name == "test_bench"
        assert len(br.level_results) == 2
        assert br.aggregate_score == pytest.approx(15.0)

    def test_empty_level_results_allowed(self) -> None:
        br = BenchmarkResult(
            benchmark_name="empty",
            level_results=[],
            aggregate_score=0.0,
        )
        assert br.level_results == []


# ---------------------------------------------------------------------------
# Benchmark Protocol (runtime_checkable)
# ---------------------------------------------------------------------------


class TestBenchmarkProtocol:
    """Benchmark is a runtime-checkable Protocol."""

    def test_non_conforming_object_fails(self) -> None:
        assert not isinstance("not a benchmark", Benchmark)

    def test_conforming_class_passes(self) -> None:
        """A class that implements all required methods satisfies the Protocol."""

        class _MockBenchmark:
            @property
            def name(self) -> str:
                return "mock"

            @property
            def num_players(self) -> int:
                return 1

            def levels(self) -> list[BenchmarkLevel]:
                return []

            def score_level(
                self, bench_level: BenchmarkLevel, items_mined: dict
            ) -> float:
                return 0.0

            def score(self, level_results: list[LevelResult]) -> float:
                return 0.0

        assert isinstance(_MockBenchmark(), Benchmark)
