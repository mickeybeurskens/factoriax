"""Low-level action sequences for the scripted rocket agent.

A :class:`Skill` is a short-lived object that emits one action per
tick and eventually signals completion. Skills are deliberately
simple and stateful — the higher :mod:`goals` layer decides what to
do when a skill finishes.

All skills share the signature:

    skill.step(view: WorldView) -> (Result, int | None)

Where :class:`Result` is one of ``RUNNING`` (action is valid, skill
continues), ``DONE`` (skill finished, action is ``None``), or
``FAIL`` (skill cannot continue).
"""

from __future__ import annotations

import enum

from .world_model import (
    WorldView,
    direction_toward,
    face_action,
)


class Result(enum.Enum):
    """Outcome of one :class:`Skill.step` / :class:`Goal.step` call."""

    RUNNING = "running"
    DONE = "done"
    FAIL = "fail"


StepReturn = tuple[Result, int | None]


class Skill:
    """Base class for low-level action sequences."""

    def step(self, view: WorldView) -> StepReturn:  # pragma: no cover - abstract
        raise NotImplementedError


class NavigateAdjacent(Skill):
    """Walk the player to an unoccupied tile adjacent to *target*.

    The skill replans on every step using the current walkable mask,
    so newly-placed obstacles or other moving agents can't strand it.
    Finishes (DONE) when the player's position is in the set of
    adjacent tiles, even if the target tile itself is occupied. Fails
    when there is no reachable adjacent tile.
    """

    def __init__(self, target: tuple[int, int]) -> None:
        self.target = target

    def step(self, view: WorldView) -> StepReturn:
        # Already adjacent → done.
        if self._is_adjacent(view, view.player.pos):
            return Result.DONE, None

        path = view.plan_path(self.target, allow_target_occupied=True)
        if path is None:
            return Result.FAIL, None
        if not path:
            # plan_path returned "you're already there" but we're not in
            # the adjacency set — target is a standalone unreachable tile.
            return Result.FAIL, None
        return Result.RUNNING, int(path[0])

    def _is_adjacent(self, view: WorldView, pos: tuple[int, int]) -> bool:
        if pos == self.target:
            return True
        return self.target in view.adjacent_tiles(pos)


class NavigateTo(Skill):
    """Walk the player onto *target* (target must be walkable).

    Used for actions that operate on the tile the player is standing
    on rather than the tile in front — the canonical case is
    :class:`Action.MINE`.
    """

    def __init__(self, target: tuple[int, int]) -> None:
        self.target = target

    def step(self, view: WorldView) -> StepReturn:
        if view.player.pos == self.target:
            return Result.DONE, None
        path = view.plan_path(self.target, allow_target_occupied=False)
        if path is None or not path:
            return Result.FAIL, None
        return Result.RUNNING, int(path[0])


class StandOnAndAct(Skill):
    """Walk onto *target* and emit *action* on the next tick.

    Direction-agnostic: the action fires regardless of facing. Used
    for ``MINE``, which reads the block under the player's feet.
    """

    def __init__(self, target: tuple[int, int], action: int) -> None:
        self.target = target
        self.action = int(action)
        self._navigator = NavigateTo(target)
        self._fired = False

    def step(self, view: WorldView) -> StepReturn:
        if self._fired:
            return Result.DONE, None
        if view.player.pos != self.target:
            return self._navigator.step(view)
        self._fired = True
        return Result.RUNNING, self.action


class FaceAndInteract(Skill):
    """Stand adjacent to *target*, face it, then emit *interact_action*.

    Composes :class:`NavigateAdjacent` with a facing step and a single
    interaction. The interaction can be any action whose target is the
    tile in front of the player: ``MINE``, ``PLACE_<machine>``,
    ``WITHDRAW_<item>``, ``DEPOSIT_<item>``.
    """

    def __init__(self, target: tuple[int, int], interact_action: int) -> None:
        self.target = target
        self.interact_action = int(interact_action)
        self._navigator = NavigateAdjacent(target)
        self._fired = False

    def step(self, view: WorldView) -> StepReturn:
        if self._fired:
            return Result.DONE, None

        # Not yet adjacent — let the navigator drive.
        if view.player.pos != self.target and not self._is_adjacent(view):
            return self._navigator.step(view)

        # Adjacent. Make sure we're facing the target.
        desired_dir = direction_toward(view.player.pos, self.target)
        if desired_dir is None:
            # Standing on the target itself; most interactions (mine,
            # place, …) fire into the tile in front, so we can't act on
            # the current tile. Step off and re-approach.
            return Result.FAIL, None

        if view.player.direction != desired_dir:
            return Result.RUNNING, face_action(desired_dir)

        # Facing the right way — emit the interaction. The skill is
        # done after this action regardless of whether it "succeeded"
        # inside the env; the goal layer owns retries.
        self._fired = True
        return Result.RUNNING, self.interact_action

    def _is_adjacent(self, view: WorldView) -> bool:
        return self.target in view.adjacent_tiles(view.player.pos)


class EmitOnce(Skill):
    """Emit a single action on the next tick, then report DONE.

    Useful for direction-agnostic actions like ``CRAFT_*`` and
    ``RESEARCH_*`` that don't need navigation or facing.
    """

    def __init__(self, action: int) -> None:
        self.action = int(action)
        self._fired = False

    def step(self, view: WorldView) -> StepReturn:  # noqa: ARG002
        if self._fired:
            return Result.DONE, None
        self._fired = True
        return Result.RUNNING, self.action
