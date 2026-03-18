"""Tests for benchmarks.runner: BenchmarkRunner validation and execution.

A session-scoped runner and stub level are shared across all execution
tests so JIT compiles once for the 10×10 shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.benchmarks.core import BenchmarkLevel, LevelResult
from factoriax.benchmarks.runner import BenchmarkRunner
from factoriax.constants import BlockType
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams

# ---------------------------------------------------------------------------
# Shared stub infrastructure
# ---------------------------------------------------------------------------


def _stub_level(name: str = "stub", max_timesteps: int = 5) -> BenchmarkLevel:
    level = LevelBuilder(10, 10).fill_rect(2, 2, 3, 3, BlockType.COAL).build(name)
    params = EnvParams(
        map_width=10, map_height=10, num_players=1, max_timesteps=max_timesteps
    )
    return BenchmarkLevel(
        name=name, description="Stub.", level=level, env_params=params
    )


class _StubBenchmark:
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


@pytest.fixture(scope="session")
def runner() -> BenchmarkRunner:
    """Shared runner — JIT compiles the 10×10 shape once per session."""
    return BenchmarkRunner(seed=0)


@pytest.fixture(scope="session")
def noop_result(runner: BenchmarkRunner):
    """Single shared run used by all execution assertions."""
    bench = _StubBenchmark(_stub_level())
    return runner.run(bench, policies=[lambda obs: jnp.array(0)])


# ---------------------------------------------------------------------------
# Validation — separate runners needed to test error paths
# ---------------------------------------------------------------------------


class TestRunnerValidation:
    def test_wrong_policy_count_raises(self) -> None:
        bench = _StubBenchmark(_stub_level(), num_players=1)
        with pytest.raises(ValueError, match="1 policy"):
            BenchmarkRunner(seed=0).run(bench, policies=[])

    def test_two_players_wrong_count_raises(self) -> None:
        bench = _StubBenchmark(_stub_level(), num_players=2)
        with pytest.raises(ValueError, match="2 policies"):
            BenchmarkRunner(seed=0).run(bench, policies=[lambda obs: jnp.array(0)])

    def test_correct_count_does_not_raise(self, runner: BenchmarkRunner) -> None:
        bench = _StubBenchmark(_stub_level(), num_players=1)
        assert runner.run(bench, policies=[lambda obs: jnp.array(0)]) is not None


# ---------------------------------------------------------------------------
# Execution — all assertions use the shared noop_result
# ---------------------------------------------------------------------------


class TestRunnerExecution:
    def test_benchmark_name(self, noop_result) -> None:
        assert noop_result.benchmark_name == "stub_benchmark"

    def test_one_level_result(self, noop_result) -> None:
        assert len(noop_result.level_results) == 1

    def test_level_result_name(self, noop_result) -> None:
        assert noop_result.level_results[0].level_name == "stub"

    def test_timesteps_at_most_max(self, noop_result) -> None:
        assert noop_result.level_results[0].timesteps_used <= 5

    def test_actions_length_matches_timesteps(self, noop_result) -> None:
        lr = noop_result.level_results[0]
        assert lr.actions.shape == (lr.timesteps_used,)

    def test_items_mined_keys(self, noop_result) -> None:
        assert set(noop_result.level_results[0].items_mined.keys()) == {
            "coal",
            "iron",
            "copper",
        }

    def test_items_mined_non_negative(self, noop_result) -> None:
        assert all(v >= 0 for v in noop_result.level_results[0].items_mined.values())

    def test_aggregate_equals_benchmark_score(self, noop_result) -> None:
        bench = _StubBenchmark(_stub_level())
        expected = bench.score(noop_result.level_results)
        assert noop_result.aggregate_score == pytest.approx(expected)

    def test_mine_beats_noop(self, runner: BenchmarkRunner) -> None:
        bench = _StubBenchmark(_stub_level(max_timesteps=20))
        noop = runner.run(bench, policies=[lambda obs: jnp.array(0)])
        mine = runner.run(bench, policies=[lambda obs: jnp.array(5)])
        assert (
            mine.level_results[0].items_mined["coal"]
            >= noop.level_results[0].items_mined["coal"]
        )

    def test_reproducible_with_same_seed(self) -> None:
        bench = _StubBenchmark(_stub_level(max_timesteps=10))
        key = jax.random.PRNGKey(7)

        def _policy(obs):
            nonlocal key
            key, subkey = jax.random.split(key)
            return jax.random.randint(subkey, shape=(), minval=0, maxval=12)

        key = jax.random.PRNGKey(7)
        r1 = BenchmarkRunner(seed=0).run(bench, policies=[_policy])
        key = jax.random.PRNGKey(7)
        r2 = BenchmarkRunner(seed=0).run(bench, policies=[_policy])
        assert r1.level_results[0].items_mined == r2.level_results[0].items_mined
        np.testing.assert_array_equal(
            r1.level_results[0].actions, r2.level_results[0].actions
        )
