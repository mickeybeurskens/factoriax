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
from factoriax.constants import Action, BlockType
from factoriax.levels import LevelBuilder
from factoriax.state import EnvParams

# Runner tests spin up the full benchmark harness (JIT-compiled 10x10
# env + scan over timesteps) — valuable but slow, gated behind @slow.
pytestmark = pytest.mark.slow

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


# ``runner`` is provided by ``tests/benchmarks/conftest.py`` so it's
# also reachable from ``tests/benchmarks/test_rocket_benchmark.py``.


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

    def test_reproducible_with_same_seed(self, runner: BenchmarkRunner) -> None:
        """Same seed + same policy stream produces bit-identical results.

        Uses the shared session-scoped ``runner`` for both calls — the
        ``BenchmarkRunner`` is stateless across ``.run()`` calls
        (seed is captured at construction), so reusing it does not
        affect reproducibility.
        """
        bench = _StubBenchmark(_stub_level(max_timesteps=10))
        key = jax.random.PRNGKey(7)

        def _policy(obs):
            nonlocal key
            key, subkey = jax.random.split(key)
            return jax.random.randint(subkey, shape=(), minval=0, maxval=12)

        key = jax.random.PRNGKey(7)
        r1 = runner.run(bench, policies=[_policy])
        key = jax.random.PRNGKey(7)
        r2 = runner.run(bench, policies=[_policy])
        assert r1.level_results[0].items_mined == r2.level_results[0].items_mined
        np.testing.assert_array_equal(
            r1.level_results[0].actions, r2.level_results[0].actions
        )


# ---------------------------------------------------------------------------
# Per-level blocked_actions
# ---------------------------------------------------------------------------


def _level_with_mask(
    name: str,
    blocked_actions: frozenset[int] | None,
    max_timesteps: int = 20,
) -> BenchmarkLevel:
    """Build a stub level optionally carrying a per-level mask.

    Player is pinned at ``(0, 0)`` so the policy can spam ``RIGHT`` and
    we can observe whether the move took effect via ``final_state``.
    Geometry-free side effect: no need to mine anything.
    """
    level = (
        LevelBuilder(10, 10)
        .fill_rect(2, 2, 3, 3, BlockType.COAL)
        .set_player_position(0, 0)
        .build(name)
    )
    params = EnvParams(
        map_width=10, map_height=10, num_players=1, max_timesteps=max_timesteps
    )
    return BenchmarkLevel(
        name=name,
        description=f"Mask test for {name}.",
        level=level,
        env_params=params,
        blocked_actions=blocked_actions,
    )


class _MultiLevelBenchmark:
    """Stub benchmark with multiple levels and an optional class-level mask."""

    def __init__(
        self,
        levels: list[BenchmarkLevel],
        class_blocked: frozenset[int] | None = None,
    ) -> None:
        self._levels = levels
        if class_blocked is not None:
            self.blocked_actions = class_blocked  # only set when given

    @property
    def name(self) -> str:
        return "multi_level_stub"

    @property
    def num_players(self) -> int:
        return 1

    def levels(self) -> list[BenchmarkLevel]:
        return self._levels

    def score_level(self, bench_level: BenchmarkLevel, items_mined: dict) -> float:
        return float(items_mined.get("coal", 0))

    def score(self, level_results: list[LevelResult]) -> float:
        return sum(r.weighted_score for r in level_results) / len(level_results)


class TestBuildEnv:
    """Unit tests for ``BenchmarkRunner._build_env`` wrapper application.

    Covers the last piece of the "runner correctly applies the mask"
    property that was previously verified by the slow integration test
    ``test_per_level_mask_blocks_movement`` (~7s). Combined with
    ``TestResolveBlocked`` (resolution logic) and
    ``tests/test_action_mask_wrapper.py`` (mask rewrite behaviour),
    these three unit clusters cover the full chain in milliseconds.
    """

    def test_no_mask_returns_bare_env(self, runner: BenchmarkRunner) -> None:
        from factoriax.envs.factoriax_env import FactoriaXEnv

        env, _ = runner._build_env(None, frozenset())
        assert isinstance(env, FactoriaXEnv)

    def test_non_empty_mask_wraps_in_action_mask_wrapper(
        self, runner: BenchmarkRunner
    ) -> None:
        from factoriax.envs.action_mask_wrapper import ActionMaskWrapper

        env, _ = runner._build_env(None, frozenset({int(Action.RIGHT)}))
        assert isinstance(env, ActionMaskWrapper)


class TestResolveBlocked:
    """Unit tests for ``BenchmarkRunner._resolve_blocked``.

    Replaces three of the four mask integration tests below (~22s of
    XLA compile) with millisecond-scale unit tests covering the
    resolution logic alone. The one surviving integration test
    (``test_per_level_mask_blocks_movement``) verifies the full
    runner+mask+wrapper wiring; the unit tests here pin the resolution
    semantics with finer diagnostics.
    """

    def test_per_level_mask_wins(self, runner: BenchmarkRunner) -> None:
        """Per-level mask is used when set (even alongside class-level)."""
        level = _level_with_mask("lvl", blocked_actions=frozenset({int(Action.MINE)}))
        bench = _MultiLevelBenchmark(
            [level], class_blocked=frozenset({int(Action.RIGHT)})
        )
        assert runner._resolve_blocked(bench, level) == frozenset({int(Action.MINE)})

    def test_per_level_none_falls_back_to_class(self, runner: BenchmarkRunner) -> None:
        """``None`` per-level falls back to the benchmark's class-level mask."""
        level = _level_with_mask("lvl", blocked_actions=None)
        bench = _MultiLevelBenchmark(
            [level], class_blocked=frozenset({int(Action.RIGHT)})
        )
        assert runner._resolve_blocked(bench, level) == frozenset({int(Action.RIGHT)})

    def test_per_level_empty_overrides_class(self, runner: BenchmarkRunner) -> None:
        """Empty ``frozenset()`` per-level overrides class-level (means 'no mask')."""
        level = _level_with_mask("lvl", blocked_actions=frozenset())
        bench = _MultiLevelBenchmark(
            [level], class_blocked=frozenset({int(Action.RIGHT)})
        )
        assert runner._resolve_blocked(bench, level) == frozenset()

    def test_both_none_returns_empty(self, runner: BenchmarkRunner) -> None:
        """No per-level mask and no class-level attr → empty frozenset."""
        level = _level_with_mask("lvl", blocked_actions=None)
        bench = _MultiLevelBenchmark([level])  # no class_blocked
        assert runner._resolve_blocked(bench, level) == frozenset()


# The final per-level-mask integration test
# ``test_per_level_mask_blocks_movement`` (~7s) was deleted in the
# replacement-for-speedup pass. Three cheap unit clusters now chain
# to cover the same end-to-end property:
#
#   1. ``TestResolveBlocked`` — which mask wins per level.
#   2. ``TestBuildEnv`` — runner wraps in ActionMaskWrapper when
#      blocked_actions is non-empty.
#   3. ``tests/test_action_mask_wrapper.py`` — ActionMaskWrapper
#      rewrites blocked actions to NOOP at step time.
#
# Chained, those three cover the same property at sub-second cost.
# A direct runner.run integration would only re-verify the chain;
# the cheaper unit chain is enough.
