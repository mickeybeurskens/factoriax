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

Debug log (optional): set ``FACTORIAX_GOAL_LOG=1`` in the
environment or pass ``debug_log=True`` to the ``Planner`` constructor
to record every goal transition as ``(tick, event, goal_name)``.
Dump via :meth:`Planner.goal_log_lines`.
"""

from __future__ import annotations

import dataclasses
import os

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


class Planner:
    """Sequential planner over a list of :class:`Goal` objects."""

    def __init__(
        self,
        goals: list[Goal],
        fail_retries: int = 3,
        *,
        debug_log: bool | None = None,
    ) -> None:
        self._queue: list[_PhaseEntry] = [
            _PhaseEntry(g, fail_retries=fail_retries) for g in goals
        ]
        self._current: _PhaseEntry | None = None
        self._tick = 0
        env_debug = os.environ.get(_ENV_DEBUG, "") == "1"
        self._debug = env_debug if debug_log is None else debug_log
        # Events: list of (tick, event_kind, goal_name). Kinds:
        # START (goal begins), DONE (completed), FAIL_RETRY
        # (retry after FAIL), FAIL_GIVEUP (exhausted retries).
        self._events: list[tuple[int, str, str]] = []

    @property
    def is_done(self) -> bool:
        return self._current is None and not self._queue

    def goal_log_lines(self) -> list[str]:
        """Return the logged goal transitions as human-readable lines."""
        return [f"{t:>6}  {kind:<12}  {name}" for t, kind, name in self._events]

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
            its retry budget. Otherwise ``(Result.RUNNING, action)``.
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
            self._log("DONE", self._current.goal.name)
            self._current = None
            return Result.RUNNING, int(Action.NOOP)

        if result is Result.FAIL:
            self._current._retries_used += 1
            if self._current._retries_used >= self._current.fail_retries:
                self._log("FAIL_GIVEUP", self._current.goal.name)
                self._current = None
                return Result.RUNNING, int(Action.NOOP)
            self._log("FAIL_RETRY", self._current.goal.name)
            return Result.RUNNING, int(Action.NOOP)

        return Result.RUNNING, action

    def _is_finished(self) -> bool:
        """True when the current entry has no more retries to offer."""
        return (
            self._current is not None
            and self._current._retries_used >= self._current.fail_retries
        )
