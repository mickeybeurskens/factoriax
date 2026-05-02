"""Phase-list planner for the scripted rocket agent.

The planner owns a sequential list of goals. On every tick it:

1. Asks the current goal for an action.
2. If the goal reports DONE, calls its :meth:`Goal.verify` post-condition.
   On a verify miss the planner consults the goal's
   :attr:`Goal.verify_failure_action` to decide whether to halt the
   run, retry, or advance as if DONE.
3. If the goal reports DONE *and* verify passes, drops the goal and
   advances to the next one.
4. If the goal reports FAIL, retries it a bounded number of times
   before giving up and moving on — many failures (e.g. ingredient
   missing) self-heal on later ticks when upstream goals top up
   inventory.

Simpler than a general behaviour tree but plenty for the rocket
objective.  If a future task demands branching logic, swap this
planner out without touching the skill/goal layers.

Debug log (optional): set ``FACTORIAX_GOAL_LOG=1`` in the
environment or pass ``debug_log=True`` to the ``Planner`` constructor
to record every goal transition as ``(tick, event, goal_name)``.
Dump via :meth:`Planner.goal_log_lines`.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

from factoriax.constants import Action

from .goals import Goal
from .skills import Result, StepReturn
from .world_model import WorldView

_ENV_DEBUG = "FACTORIAX_GOAL_LOG"


@dataclasses.dataclass
class _PhaseEntry:
    """Bookkeeping for one goal in the phase list."""

    goal: Goal
    fail_retries: int = 3
    _retries_used: int = 0


@dataclass(frozen=True)
class VerifyDiagnostic:
    """Structured record of the most recent verify failure.

    Attributes:
        tick: Planner tick on which the failure was detected.
        goal_name: ``Goal.name`` of the offending goal.
        goal_repr: ``repr(goal)`` for finer-grained identification.
        action_taken: Which branch of
            :data:`~baselines.rocket.scripted.goals.VerifyFailureAction`
            ran in response (``"halt"`` | ``"retry"`` | ``"ignore"``).
        details: Optional multi-line explanation produced by the goal
            via :meth:`~baselines.rocket.scripted.goals.Goal.verify_failure_details`.
            ``None`` if the goal didn't override the hook.
    """

    tick: int
    goal_name: str
    goal_repr: str
    action_taken: str
    details: str | None = None

    def format(self) -> str:
        """Render the diagnostic as a multi-line, human-readable block.

        Top line gives the tick + the goal repr; subsequent lines hold
        the goal-supplied details (when present). Suitable for
        printing to a console or attaching to a log artifact.
        """
        head = (
            f"VERIFY_FAIL at tick={self.tick} ({self.action_taken})\n"
            f"  goal: {self.goal_repr}"
        )
        if self.details:
            return f"{head}\n{self.details}"
        return head


class Planner:
    """Sequential planner over a list of :class:`Goal` objects."""

    def __init__(
        self,
        goals: list[Goal],
        fail_retries: int = 3,
        *,
        debug_log: bool | None = None,
        halt_on_verify_fail: bool = True,
    ) -> None:
        """Construct the planner.

        Args:
            goals: Goals to execute in order.
            fail_retries: Per-goal step-FAIL retry budget. Verify
                failures with ``verify_failure_action="retry"`` also
                draw from this budget.
            debug_log: Force-enable goal-transition logging. ``None``
                (default) reads the ``FACTORIAX_GOAL_LOG`` env var.
            halt_on_verify_fail: When ``True`` (default), a goal whose
                verify returns ``False`` *and* whose
                ``verify_failure_action`` is ``"halt"`` returns
                :attr:`Result.FAIL` from :meth:`step`, propagating to
                the run loop. When ``False``, the planner logs the
                failure and advances as if DONE — useful for
                benchmark runs where we want to score whatever was
                achieved without aborting the episode.
        """
        self._queue: list[_PhaseEntry] = [
            _PhaseEntry(g, fail_retries=fail_retries) for g in goals
        ]
        self._current: _PhaseEntry | None = None
        self._tick = 0
        env_debug = os.environ.get(_ENV_DEBUG, "") == "1"
        self._debug = env_debug if debug_log is None else debug_log
        self._halt_on_verify_fail = halt_on_verify_fail
        # Events: list of (tick, event_kind, goal_name). Kinds:
        # START (goal begins), DONE (completed), FAIL_RETRY
        # (retry after FAIL), FAIL_GIVEUP (exhausted retries),
        # VERIFY_HALT (verify failed, halt mode), VERIFY_RETRY
        # (verify failed, retry budget consumed), VERIFY_IGNORE
        # (verify failed, advanced anyway).
        self._events: list[tuple[int, str, str]] = []
        self._verify_diagnostic: VerifyDiagnostic | None = None

    @property
    def is_done(self) -> bool:
        """``True`` once every goal has either completed or been retired."""
        return self._current is None and not self._queue

    @property
    def verify_diagnostic(self) -> VerifyDiagnostic | None:
        """Most recent :class:`VerifyDiagnostic`, or ``None`` if no failure."""
        return self._verify_diagnostic

    def goal_log_lines(self) -> list[str]:
        """Return the logged goal transitions as human-readable lines."""
        return [f"{t:>6}  {kind:<14}  {name}" for t, kind, name in self._events]

    @property
    def current_goal_name(self) -> str:
        """Name of the goal currently running (or ``""`` when idle)."""
        return self._current.goal.name if self._current is not None else ""

    def _log(self, kind: str, name: str) -> None:
        if self._debug:
            self._events.append((self._tick, kind, name))

    def step(self, view: WorldView) -> StepReturn:
        """Advance the plan by one tick.

        Returns:
            ``(Result.DONE, None)`` when every queued goal has
            completed. ``(Result.FAIL, None)`` when a goal exhausted
            its retry budget *or* a verify failure with action
            ``"halt"`` fired (and ``halt_on_verify_fail`` is enabled).
            Otherwise ``(Result.RUNNING, action)``.
        """
        self._tick += 1

        while self._current is None or self._is_finished():
            if not self._queue:
                return Result.DONE, None
            self._current = self._queue.pop(0)
            self._log("START", self._current.goal.name)

        assert self._current is not None
        result, action = self._current.goal.step(view)

        if result is Result.DONE:
            return self._on_step_done(view)

        if result is Result.FAIL:
            self._current._retries_used += 1
            if self._current._retries_used >= self._current.fail_retries:
                self._log("FAIL_GIVEUP", self._current.goal.name)
                self._current = None
                return Result.RUNNING, int(Action.NOOP)
            self._log("FAIL_RETRY", self._current.goal.name)
            return Result.RUNNING, int(Action.NOOP)

        return Result.RUNNING, action

    def _on_step_done(self, view: WorldView) -> StepReturn:
        """Handle a goal whose :meth:`step` returned ``DONE``.

        Runs the goal's :meth:`verify` post-condition; on success the
        goal is retired. On failure the goal's
        :attr:`verify_failure_action` decides whether the planner
        halts, retries, or ignores.
        """
        assert self._current is not None
        goal = self._current.goal
        if goal.verify(view):
            self._log("DONE", goal.name)
            self._current = None
            return Result.RUNNING, int(Action.NOOP)

        action = goal.verify_failure_action
        self._verify_diagnostic = VerifyDiagnostic(
            tick=self._tick,
            goal_name=goal.name,
            goal_repr=repr(goal),
            action_taken=action,
            details=goal.verify_failure_details(view),
        )

        if action == "halt" and self._halt_on_verify_fail:
            self._log("VERIFY_HALT", goal.name)
            self._current = None
            return Result.FAIL, None

        if action == "retry":
            self._current._retries_used += 1
            if self._current._retries_used >= self._current.fail_retries:
                self._log("VERIFY_GIVEUP", goal.name)
                self._current = None
                return Result.RUNNING, int(Action.NOOP)
            self._log("VERIFY_RETRY", goal.name)
            return Result.RUNNING, int(Action.NOOP)

        # action == "ignore", or "halt" with halt_on_verify_fail=False.
        self._log("VERIFY_IGNORE", goal.name)
        self._current = None
        return Result.RUNNING, int(Action.NOOP)

    def _is_finished(self) -> bool:
        """True when the current entry has no more retries to offer."""
        return (
            self._current is not None
            and self._current._retries_used >= self._current.fail_retries
        )
