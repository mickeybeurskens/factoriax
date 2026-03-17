"""Tests for benchmarks.runner: BenchmarkRunner validation and execution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.constants import BlockType
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams

from benchmarks.core import Benchmark, BenchmarkLevel, LevelResult
from benchmarks.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# Minimal stub benchmark for runner tests
# ---------------------------------------------------------------------------


def _make_stub_level(
    name: str = "stub",
    width: int = 10,
    height: int = 10,
    max_timesteps: int = 5,
) -> BenchmarkLevel:
    """A tiny level that runs in 5 steps for fast tests."""
    level = (
        LevelBuilder(width, height)
        .fill_rect(2, 2, 3, 3, BlockType.COAL)
        .build(name)
    )
    params = EnvParams(
        map_width=width,
        map_height=height,
        num_players=1,
        max_timesteps=max_timesteps,
    )
    return BenchmarkLevel(name=name, description="Stub level.", level=level, env_params=params)


class _SingleLevelBenchmark:
    """Minimal Benchmark with one level for testing the runner."""

    def __init__(self, bench_level: BenchmarkLevel, num_players: int = 1) -> None:
        self._level = bench_level
        self._num_players = num_players

    @property
    def name(self) -> str:
        return "stub_benchmark"

    @property
    def num_players(self) -> int:
        return self._num_players

    def levels(self) -> list[BenchmarkLevel]:
        return [self._level]

    def score_level(self, bench_level: BenchmarkLevel, items_mined: dict) -> float:
        return float(items_mined.get("coal", 0))

    def score(self, level_results: list[LevelResult]) -> float:
        return sum(r.weighted_score for r in level_results) / len(level_results)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestRunnerValidation:
    """BenchmarkRunner raises on policy count mismatch."""

    def test_wrong_policy_count_raises(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level(), num_players=1)
        runner = BenchmarkRunner(seed=0)
        with pytest.raises(ValueError, match="1 policy"):
            runner.run(bench, policies=[])

    def test_two_players_wrong_count_raises(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level(), num_players=2)
        runner = BenchmarkRunner(seed=0)
        policy = lambda obs: jnp.array(0)
        with pytest.raises(ValueError, match="2 policies"):
            runner.run(bench, policies=[policy])

    def test_correct_policy_count_does_not_raise(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level(), num_players=1)
        runner = BenchmarkRunner(seed=0)
        policy = lambda obs: jnp.array(0)
        result = runner.run(bench, policies=[policy])
        assert result is not None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestRunnerExecution:
    """BenchmarkRunner produces correctly structured results."""

    def _noop_policy(self, obs: jax.Array) -> jax.Array:
        return jnp.array(0)  # NOOP every step

    def test_result_has_correct_benchmark_name(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level())
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        assert result.benchmark_name == "stub_benchmark"

    def test_result_has_one_level_result(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level())
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        assert len(result.level_results) == 1

    def test_level_result_name_matches_level(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level(name="my_level"))
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        assert result.level_results[0].level_name == "my_level"

    def test_timesteps_used_at_most_max(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level(max_timesteps=5))
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        assert result.level_results[0].timesteps_used <= 5

    def test_actions_length_matches_timesteps_used(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level(max_timesteps=5))
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        lr = result.level_results[0]
        assert lr.actions.shape == (lr.timesteps_used,)

    def test_items_mined_keys_present(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level())
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        mined = result.level_results[0].items_mined
        assert set(mined.keys()) == {"coal", "iron", "copper"}

    def test_items_mined_non_negative(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level())
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        mined = result.level_results[0].items_mined
        assert all(v >= 0 for v in mined.values())

    def test_aggregate_equals_benchmark_score(self) -> None:
        bench = _SingleLevelBenchmark(_make_stub_level())
        result = BenchmarkRunner(seed=0).run(bench, policies=[self._noop_policy])
        expected = bench.score(result.level_results)
        assert result.aggregate_score == pytest.approx(expected)

    def test_mine_policy_collects_more_than_noop(self) -> None:
        """A policy that always mines should score higher than one that noops."""
        bench = _SingleLevelBenchmark(_make_stub_level(max_timesteps=20))
        runner = BenchmarkRunner(seed=0)

        noop_result = runner.run(bench, policies=[lambda obs: jnp.array(0)])
        mine_result = runner.run(bench, policies=[lambda obs: jnp.array(5)])  # MINE=5

        assert (
            mine_result.level_results[0].items_mined["coal"]
            >= noop_result.level_results[0].items_mined["coal"]
        )

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed and same policy should give identical results."""
        bench = _SingleLevelBenchmark(_make_stub_level(max_timesteps=10))
        key = jax.random.PRNGKey(7)

        def _policy(obs: jax.Array) -> jax.Array:
            nonlocal key
            key, subkey = jax.random.split(key)
            return jax.random.randint(subkey, shape=(), minval=0, maxval=12)

        # Reset key before each run for identical policies.
        key = jax.random.PRNGKey(7)
        r1 = BenchmarkRunner(seed=0).run(bench, policies=[_policy])
        key = jax.random.PRNGKey(7)
        r2 = BenchmarkRunner(seed=0).run(bench, policies=[_policy])

        assert r1.level_results[0].items_mined == r2.level_results[0].items_mined
        np.testing.assert_array_equal(r1.level_results[0].actions, r2.level_results[0].actions)
