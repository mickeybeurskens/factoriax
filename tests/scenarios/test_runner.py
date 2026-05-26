"""Tests for scenarios.runner: ScenarioRunner validation and execution.

The runner's contract is to thread policies through env steps,
accumulate per-level results, validate policy counts, resolve masks,
and aggregate scores. Env semantics (what MINE does, how mining
accumulates, JAX determinism guarantees) belong to the env and
scenario layers, not to the runner. These tests use a ``_StubRunner``
that synthesises ``LevelResult`` instances without triggering any
``factoriax_step`` XLA compile — runner-specific properties stay
covered, env-specific ones move to env/scenario tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import Action, BlockType
from factoriax.engine.levels import LevelBuilder
from factoriax.engine.state import EnvParams
from factoriax.scenarios.core import LevelResult, ScenarioLevel
from factoriax.scenarios.runner import ScenarioRunner

# ---------------------------------------------------------------------------
# Shared stub infrastructure
# ---------------------------------------------------------------------------


def _stub_level(name: str = "stub", max_timesteps: int = 5) -> ScenarioLevel:
    level = LevelBuilder(10, 10).fill_rect(2, 2, 3, 3, BlockType.COAL).build(name)
    params = EnvParams(
        map_width=10, map_height=10, num_players=1, max_timesteps=max_timesteps
    )
    return ScenarioLevel(name=name, description="Stub.", level=level, env_params=params)


class _StubScenario:
    def __init__(self, scenario_level: ScenarioLevel, num_players: int = 1) -> None:
        self._level = scenario_level
        self._num_players = num_players

    @property
    def name(self) -> str:
        return "stub_scenario"

    @property
    def num_players(self) -> int:
        return self._num_players

    def levels(self) -> list[ScenarioLevel]:
        return [self._level]

    def score_level(self, scenario_level: ScenarioLevel, items_mined: dict) -> float:
        return float(items_mined.get("coal", 0))

    def score(self, level_results: list[LevelResult]) -> float:
        return sum(r.weighted_score for r in level_results) / len(level_results)


class _StubRunner(ScenarioRunner):
    """Runner that synthesises ``LevelResult`` without env compile.

    Overrides the two hooks that touch the env: ``_ensure_env`` becomes
    a no-op (no FactoriaXEnv built, no ``jax.jit(step_env)`` cached),
    and ``_run_level`` returns a deterministic LevelResult by polling
    each policy ``max_timesteps`` times on a sentinel observation —
    enough to verify the runner's looping, scoring, and aggregation
    contract without paying ~7s for ``factoriax_step``'s XLA compile.
    """

    def _ensure_env(self, scenario: Any, scenario_level: ScenarioLevel) -> Any:
        """No env to build. Resolve achievement_fn for ``_current_fn`` parity."""
        fn = getattr(scenario, "achievement_fn", None)
        self._current_fn = fn
        return fn

    def _run_level(
        self,
        scenario: Any,
        scenario_level: ScenarioLevel,
        policies: list[Callable[[jax.Array], jax.Array]],
        rng: jax.Array,
        obs_fn: Any,
        constraint_fn: Any,
        achievement_fn: Any,
    ) -> LevelResult:
        """Poll the policy for ``max_timesteps`` actions and pack a result.

        ``items_mined`` is always zeroed — the runner doesn't own
        mining semantics, so a stub's "items mined" doesn't carry
        runner-relevant signal.
        """
        del obs_fn, constraint_fn, achievement_fn  # not exercised by the stub
        params = scenario_level.env_params
        sentinel_obs = jnp.zeros(1)
        actions = np.array(
            [int(policies[0](sentinel_obs)) for _ in range(int(params.max_timesteps))],
            dtype=np.int32,
        )
        items_mined = {"coal": 0, "iron": 0, "copper": 0}
        return LevelResult(
            level_name=scenario_level.name,
            items_mined=items_mined,
            weighted_score=scenario.score_level(scenario_level, items_mined),
            timesteps_used=int(actions.shape[0]),
            actions=actions,
        )


@pytest.fixture(scope="session")
def stub_runner() -> _StubRunner:
    """Compile-free runner; safe to share across the file."""
    return _StubRunner(seed=0)


@pytest.fixture(scope="session")
def noop_result(stub_runner: _StubRunner):
    """Single shared run used by all execution assertions."""
    scenario = _StubScenario(_stub_level())
    return stub_runner.run(scenario, policies=[lambda obs: jnp.array(0)])


# ---------------------------------------------------------------------------
# Validation — separate runners needed to test error paths
# ---------------------------------------------------------------------------


class TestRunnerValidation:
    def test_wrong_policy_count_raises(self) -> None:
        scenario = _StubScenario(_stub_level(), num_players=1)
        with pytest.raises(ValueError, match="1 policy"):
            ScenarioRunner(seed=0).run(scenario, policies=[])

    def test_two_players_wrong_count_raises(self) -> None:
        scenario = _StubScenario(_stub_level(), num_players=2)
        with pytest.raises(ValueError, match="2 policies"):
            ScenarioRunner(seed=0).run(scenario, policies=[lambda obs: jnp.array(0)])

    # ``test_correct_count_does_not_raise`` was removed: the
    # ``noop_result`` fixture (TestRunnerExecution) builds a
    # ``_StubScenario`` with one player and runs it through the same
    # runner with a single policy. If that fixture's run had raised,
    # every TestRunnerExecution test below would fail at setup. The
    # dedicated "doesn't raise" assertion paid the first ~7s
    # ``step_env`` compile for the file but added nothing the fixture
    # didn't already cover.


# ---------------------------------------------------------------------------
# Execution — all assertions use the shared noop_result
# ---------------------------------------------------------------------------


class TestRunnerExecution:
    def test_scenario_name(self, noop_result) -> None:
        assert noop_result.scenario_name == "stub_scenario"

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

    def test_aggregate_equals_scenario_score(self, noop_result) -> None:
        scenario = _StubScenario(_stub_level())
        expected = scenario.score(noop_result.level_results)
        assert noop_result.aggregate_score == pytest.approx(expected)

    # ``test_mine_beats_noop`` and ``test_reproducible_with_same_seed``
    # were removed in the runner-stubbing pass. Both verified env or
    # JAX-level semantics rather than runner contract:
    #
    #   - "MINE accumulates more coal than NOOP" is a property of
    #     factoriax_step / the mining handler. Covered by
    #     tests/test_game_logic.py::test_mine_decrements_resources
    #     and the env's per-action tests in test_factoriax.py.
    #
    #   - "Same seed → same trajectory" is a JAX determinism
    #     guarantee, not a runner contract. The runner-level claim is
    #     narrower: ``run()`` derives its initial rng from ``self.seed``.
    #     That's covered below in ``TestSeedPlumbing``.


class TestSeedPlumbing:
    """The runner threads ``self.seed`` into its initial PRNGKey.

    JAX guarantees that the same PRNGKey produces the same trajectory.
    The runner-specific contract is the one-line plumbing in ``run()``:
    ``rng = jax.random.PRNGKey(self.seed)``. Two integration runs were
    previously used to verify this; a single monkeypatch suffices.
    """

    def test_run_derives_initial_rng_from_self_seed(self, monkeypatch) -> None:
        """``runner.run()`` calls ``PRNGKey(self.seed)`` exactly once."""
        seeds_seen: list[int] = []
        real_prng_key = jax.random.PRNGKey

        def _capture(seed):
            seeds_seen.append(int(seed))
            return real_prng_key(seed)

        monkeypatch.setattr(jax.random, "PRNGKey", _capture)
        runner = _StubRunner(seed=1234)
        scenario = _StubScenario(_stub_level())
        runner.run(scenario, policies=[lambda obs: jnp.array(0)])
        assert seeds_seen[0] == 1234


# ---------------------------------------------------------------------------
# Per-level blocked_actions
# ---------------------------------------------------------------------------


def _level_with_mask(
    name: str,
    blocked_actions: frozenset[int] | None,
    max_timesteps: int = 20,
) -> ScenarioLevel:
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
    return ScenarioLevel(
        name=name,
        description=f"Mask test for {name}.",
        level=level,
        env_params=params,
        blocked_actions=blocked_actions,
    )


class _MultiLevelBenchmark:
    """Stub scenario with multiple levels and an optional class-level mask."""

    def __init__(
        self,
        levels: list[ScenarioLevel],
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

    def levels(self) -> list[ScenarioLevel]:
        return self._levels

    def score_level(self, scenario_level: ScenarioLevel, items_mined: dict) -> float:
        return float(items_mined.get("coal", 0))

    def score(self, level_results: list[LevelResult]) -> float:
        return sum(r.weighted_score for r in level_results) / len(level_results)


class TestAchievementsAccessor:
    """Unit tests for ``ScenarioRunner._achievements``.

    Replaces the ~9s ``test_runner_populates_achievements_end_to_end``
    in ``test_rocket_benchmark.py``: instead of building a real
    ``RocketScenario`` and running it through the runner (unique
    config compile), exercise the accessor directly. The wiring it
    used to cover is split across:

      - This file: ``_achievements`` returns numpy of unlocks when
        an ``achievement_fn`` is bound.
      - This file: ``_ensure_env`` resolves scenario.achievement_fn
        (covered indirectly by TestBuildEnv + TestResolveBlocked).
      - ``test_rocket_benchmark.py``: RocketScenario.achievement_fn
        attribute is set to ``rocket_conditions``.
      - ``test_runner.py::TestRunnerExecution`` tests: aggregate_score
        is a finite float.
    """

    def test_returns_none_when_no_fn(self) -> None:
        """``_achievements`` returns None when the runner has no fn."""
        r = ScenarioRunner(seed=0)  # no achievement_fn
        # Build a stub state with an achievements_unlocked field.
        state = type(
            "S", (), {"achievements_unlocked": jnp.zeros(3, dtype=jnp.bool_)}
        )()
        assert r._achievements(state) is None

    def test_returns_numpy_array_when_fn_set(self) -> None:
        """``_achievements`` returns numpy of state.achievements_unlocked."""

        # achievement_fn doesn't matter for this test — just needs to be
        # something non-None so _current_fn is set after construction.
        def _fn(state):
            return state.achievements_unlocked

        r = ScenarioRunner(seed=0, achievement_fn=_fn)
        unlocks = jnp.array([True, False, True])
        state = type("S", (), {"achievements_unlocked": unlocks})()
        result = r._achievements(state)
        assert result is not None
        np.testing.assert_array_equal(result, np.array([True, False, True]))


class TestBuildEnv:
    """Unit tests for ``ScenarioRunner._build_env`` wrapper application.

    Covers the last piece of the "runner correctly applies the mask"
    property that was previously verified by the slow integration test
    ``test_per_level_mask_blocks_movement`` (~7s). Combined with
    ``TestResolveBlocked`` (resolution logic) and
    ``tests/test_action_mask_wrapper.py`` (mask rewrite behaviour),
    these three unit clusters cover the full chain in milliseconds.
    """

    def test_no_mask_returns_bare_env(self, runner: ScenarioRunner) -> None:
        from factoriax.envs.factoriax_env import FactoriaXEnv

        env, _ = runner._build_env(None, frozenset())
        assert isinstance(env, FactoriaXEnv)

    def test_non_empty_mask_wraps_in_action_mask_wrapper(
        self, runner: ScenarioRunner
    ) -> None:
        from factoriax.envs.action_mask_wrapper import ActionMaskWrapper

        env, _ = runner._build_env(None, frozenset({int(Action.RIGHT)}))
        assert isinstance(env, ActionMaskWrapper)


class TestResolveBlocked:
    """Unit tests for ``ScenarioRunner._resolve_blocked``.

    Replaces three of the four mask integration tests below (~22s of
    XLA compile) with millisecond-scale unit tests covering the
    resolution logic alone. The one surviving integration test
    (``test_per_level_mask_blocks_movement``) verifies the full
    runner+mask+wrapper wiring; the unit tests here pin the resolution
    semantics with finer diagnostics.
    """

    def test_per_level_mask_wins(self, runner: ScenarioRunner) -> None:
        """Per-level mask is used when set (even alongside class-level)."""
        level = _level_with_mask("lvl", blocked_actions=frozenset({int(Action.MINE)}))
        scenario = _MultiLevelBenchmark(
            [level], class_blocked=frozenset({int(Action.RIGHT)})
        )
        assert runner._resolve_blocked(scenario, level) == frozenset({int(Action.MINE)})

    def test_per_level_none_falls_back_to_class(self, runner: ScenarioRunner) -> None:
        """``None`` per-level falls back to the scenario's class-level mask."""
        level = _level_with_mask("lvl", blocked_actions=None)
        scenario = _MultiLevelBenchmark(
            [level], class_blocked=frozenset({int(Action.RIGHT)})
        )
        expected = frozenset({int(Action.RIGHT)})
        assert runner._resolve_blocked(scenario, level) == expected

    def test_per_level_empty_overrides_class(self, runner: ScenarioRunner) -> None:
        """Empty ``frozenset()`` per-level overrides class-level (means 'no mask')."""
        level = _level_with_mask("lvl", blocked_actions=frozenset())
        scenario = _MultiLevelBenchmark(
            [level], class_blocked=frozenset({int(Action.RIGHT)})
        )
        assert runner._resolve_blocked(scenario, level) == frozenset()

    def test_both_none_returns_empty(self, runner: ScenarioRunner) -> None:
        """No per-level mask and no class-level attr → empty frozenset."""
        level = _level_with_mask("lvl", blocked_actions=None)
        scenario = _MultiLevelBenchmark([level])  # no class_blocked
        assert runner._resolve_blocked(scenario, level) == frozenset()


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
