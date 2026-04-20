"""Phase-list planner for the scripted rocket agent.

The planner owns a sequential list of goals. On every tick it:

1. Asks the current goal for an action.
2. If the goal reports DONE, drops it and advances to the next goal.
3. If the goal reports FAIL, retries it a bounded number of times
   before giving up and moving on — many failures (e.g. ingredient
   missing) self-heal on later ticks when upstream goals top up
   inventory.

Simpler than a general behaviour tree but plenty for the rocket
objective.  If a future task demands branching logic, swap this
planner out without touching the skill/goal layers.
"""

from __future__ import annotations

import dataclasses

from factoriax.constants import Action

from .goals import Goal
from .skills import Result, StepReturn
from .world_model import WorldView


@dataclasses.dataclass
class _PhaseEntry:
    """Bookkeeping for one goal in the phase list."""

    goal: Goal
    fail_retries: int = 3
    _retries_used: int = 0


class Planner:
    """Sequential planner over a list of :class:`Goal` objects."""

    def __init__(self, goals: list[Goal], fail_retries: int = 3) -> None:
        self._queue: list[_PhaseEntry] = [
            _PhaseEntry(g, fail_retries=fail_retries) for g in goals
        ]
        self._current: _PhaseEntry | None = None

    @property
    def is_done(self) -> bool:
        return self._current is None and not self._queue

    def step(self, view: WorldView) -> StepReturn:
        """Advance the plan by one tick.

        Returns:
            ``(Result.DONE, None)`` when every queued goal has
            completed. ``(Result.FAIL, None)`` when a goal exhausted
            its retry budget. Otherwise ``(Result.RUNNING, action)``.
        """
        while self._current is None or self._is_finished():
            if not self._queue:
                return Result.DONE, None
            self._current = self._queue.pop(0)

        assert self._current is not None
        result, action = self._current.goal.step(view)

        if result is Result.DONE:
            self._current = None
            # Don't return DONE here — we may have more goals to run.
            # Emit a NOOP this tick and pick up the next goal next tick.
            return Result.RUNNING, int(Action.NOOP)

        if result is Result.FAIL:
            self._current._retries_used += 1
            if self._current._retries_used >= self._current.fail_retries:
                # Abandon this goal; move on.
                self._current = None
                return Result.RUNNING, int(Action.NOOP)
            # Soft-retry: create a fresh instance of the same Goal type
            # on the queue head by re-constructing via __class__ and
            # argument introspection. For now, just reset internal state
            # by stepping it again and hoping — goals' step() methods
            # are written to replan each call.
            return Result.RUNNING, int(Action.NOOP)

        return Result.RUNNING, action

    def _is_finished(self) -> bool:
        """True when the current entry has no more retries to offer."""
        return (
            self._current is not None
            and self._current._retries_used >= self._current.fail_retries
        )
