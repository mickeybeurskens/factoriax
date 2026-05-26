"""Tests for the verify-aware :class:`Planner`.

Covers the post-condition handling added to the planner: the
``Goal.verify`` hook, the three ``verify_failure_action`` branches
(``halt`` | ``retry`` | ``ignore``), and the
``halt_on_verify_fail`` opt-out.
"""

from __future__ import annotations

from unittest.mock import Mock

from baselines.rocket.scripted.goals import Goal, VerifyFailureAction
from baselines.rocket.scripted.planner import Planner
from baselines.rocket.scripted.skills import Result, StepReturn
from baselines.rocket.scripted.world_model import WorldView
from factoriax.engine.constants import Action


class _StubGoal(Goal):
    """Minimal goal that reports DONE on the first ``step`` call.

    The test parametrises ``verify_returns`` and
    ``verify_failure_action`` to exercise the planner's branches
    without standing up real world state.
    """

    name: str = "StubGoal"

    def __init__(
        self,
        verify_returns: bool = True,
        action: VerifyFailureAction = "halt",
    ) -> None:
        self._verify_returns = verify_returns
        self.verify_failure_action: VerifyFailureAction = action  # type: ignore[misc]
        self.step_calls: int = 0
        self.verify_calls: int = 0

    def step(self, view: WorldView) -> StepReturn:
        self.step_calls += 1
        return Result.DONE, int(Action.NOOP)

    def verify(self, view: WorldView) -> bool:
        self.verify_calls += 1
        return self._verify_returns


def _stub_view() -> WorldView:
    """Return a Mock standing in for a WorldView (no attribute access)."""
    return Mock(spec=WorldView)


class TestVerifyPasses:
    """A goal whose verify returns True flows through unchanged."""

    def test_default_verify_returns_true(self) -> None:
        """The base ``Goal.verify`` is a no-op that returns True."""
        view = _stub_view()
        assert Goal().verify(view) is True

    def test_planner_advances_when_verify_passes(self) -> None:
        """Step DONE + verify True drops the goal and the planner advances."""
        goal = _StubGoal(verify_returns=True)
        planner = Planner([goal], debug_log=True)
        view = _stub_view()

        result, _ = planner.step(view)

        assert result is Result.RUNNING
        assert planner.is_done
        assert goal.verify_calls == 1
        assert planner.verify_diagnostic is None


class TestVerifyHalt:
    """Verify-fail with action ``halt`` returns ``Result.FAIL``."""

    def test_halt_returns_fail(self) -> None:
        goal = _StubGoal(verify_returns=False, action="halt")
        planner = Planner([goal], debug_log=True)
        view = _stub_view()

        result, action = planner.step(view)

        assert result is Result.FAIL
        assert action is None

    def test_halt_records_diagnostic(self) -> None:
        goal = _StubGoal(verify_returns=False, action="halt")
        planner = Planner([goal], debug_log=True)
        view = _stub_view()

        planner.step(view)

        diag = planner.verify_diagnostic
        assert diag is not None
        assert diag.goal_name == "StubGoal"
        assert diag.action_taken == "halt"
        assert diag.tick == 1

    def test_halt_logs_event(self) -> None:
        goal = _StubGoal(verify_returns=False, action="halt")
        planner = Planner([goal], debug_log=True)
        view = _stub_view()

        planner.step(view)

        kinds = [kind for _tick, kind, _name in planner._events]
        assert "VERIFY_HALT" in kinds

    def test_halt_disabled_advances_planner(self) -> None:
        """``halt_on_verify_fail=False`` falls through to the ignore branch."""
        goal = _StubGoal(verify_returns=False, action="halt")
        planner = Planner([goal], debug_log=True, halt_on_verify_fail=False)
        view = _stub_view()

        result, _ = planner.step(view)

        assert result is Result.RUNNING
        assert planner.is_done


class TestVerifyRetry:
    """Verify-fail with action ``retry`` consumes the retry budget."""

    def test_retry_runs_step_again(self) -> None:
        """Retry leaves the goal in place; step is called again next tick."""
        goal = _StubGoal(verify_returns=False, action="retry")
        planner = Planner([goal], fail_retries=3, debug_log=True)
        view = _stub_view()

        # Tick 1 — first step+verify (fail) + retry.
        planner.step(view)
        assert goal.step_calls == 1

        # Tick 2 — step is called again (retry).
        planner.step(view)
        assert goal.step_calls == 2

    def test_retry_exhausts_then_advances(self) -> None:
        """After the retry budget is gone the planner moves on."""
        goal = _StubGoal(verify_returns=False, action="retry")
        planner = Planner([goal], fail_retries=2, debug_log=True)
        view = _stub_view()

        # Tick 1: retry (budget 1/2). Tick 2: giveup (budget 2/2,
        # goal retired). The planner is then idle; the next tick
        # would return DONE because the queue is empty.
        planner.step(view)
        result, _ = planner.step(view)

        assert result is Result.RUNNING
        assert planner.is_done
        kinds = [kind for _tick, kind, _name in planner._events]
        assert kinds.count("VERIFY_RETRY") == 1
        assert kinds.count("VERIFY_GIVEUP") == 1


class TestVerifyIgnore:
    """Verify-fail with action ``ignore`` advances as if DONE."""

    def test_ignore_advances_planner(self) -> None:
        goal = _StubGoal(verify_returns=False, action="ignore")
        planner = Planner([goal], debug_log=True)
        view = _stub_view()

        result, _ = planner.step(view)

        assert result is Result.RUNNING
        assert planner.is_done

    def test_ignore_records_diagnostic(self) -> None:
        """Diagnostic is still populated so callers can inspect it."""
        goal = _StubGoal(verify_returns=False, action="ignore")
        planner = Planner([goal], debug_log=True)
        view = _stub_view()

        planner.step(view)

        diag = planner.verify_diagnostic
        assert diag is not None
        assert diag.action_taken == "ignore"


class TestVerifyDefaultsAreUnchanged:
    """Goals that don't override verify keep their old behaviour."""

    class _NoVerifyOverrideGoal(Goal):
        """Just inherits ``verify`` — should always pass."""

        name: str = "NoVerifyOverride"

        def step(self, view: WorldView) -> StepReturn:
            return Result.DONE, int(Action.NOOP)

    def test_default_goal_passes_through(self) -> None:
        planner = Planner([self._NoVerifyOverrideGoal()], debug_log=True)
        result, _ = planner.step(_stub_view())

        assert result is Result.RUNNING
        assert planner.is_done
        assert planner.verify_diagnostic is None
