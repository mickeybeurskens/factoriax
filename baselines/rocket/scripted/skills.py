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
    _DIR_OFFSETS,
    _DIR_TO_MOVE_ACTION,
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

        path = view.plan_path(self.target, mode="adjacent_only")
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
        path = view.plan_path(self.target, mode="exactly_on")
        if path is None or not path:
            return Result.FAIL, None
        return Result.RUNNING, int(path[0])


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

        # Standing on the target itself: most interactions (mine, place, …)
        # fire into the tile in front, so the player must step off first.
        # Try each cardinal direction; the first walkable non-target tile
        # wins. We swap the navigator to a fresh NavigateAdjacent below
        # so the next tick can re-plan from the new position.
        if view.player.pos == self.target:
            step_action = self._step_off_action(view)
            if step_action is None:
                return Result.FAIL, None
            self._navigator = NavigateAdjacent(self.target)
            return Result.RUNNING, step_action

        # Not yet adjacent — let the navigator drive.
        if not self._is_adjacent(view):
            return self._navigator.step(view)

        # Adjacent. Make sure we're facing the target.
        desired_dir = direction_toward(view.player.pos, self.target)
        if desired_dir is None:
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

    def _step_off_action(self, view: WorldView) -> int | None:
        """Return a MOVE_* action that steps the player off ``self.target``.

        Picks the first walkable cardinal neighbour that is not the target
        tile. Returns ``None`` when no such tile exists (rare: target is
        surrounded by water / out-of-bounds).
        """
        px, py = view.player.pos
        h, w = view.shape
        for direction, (dx, dy) in _DIR_OFFSETS.items():
            nx, ny = px + dx, py + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) == self.target:
                continue
            if bool(view.walkable[ny, nx]):
                return _DIR_TO_MOVE_ACTION[direction]
        return None


class PlaceAt(Skill):
    """Place a machine at *target* with an explicit facing direction.

    The machine's ``ent_direction`` is set to the player's direction
    at the moment of placement (see
    :func:`factoriax.placement.place_machine`). So to land a machine
    at ``target`` facing direction ``D``, the player must stand at
    ``target - unit_vec(D)`` and emit ``PLACE_*`` while facing ``D``.

    This matters for miners — their per-tick push is directed by
    ``ent_direction``. A miner placed without direction control may
    push into a dead tile instead of the intended pallet.
    """

    def __init__(
        self,
        target: tuple[int, int],
        direction: int,
        place_action: int,
    ) -> None:
        from .world_model import _DIR_OFFSETS  # lazy: avoids import cycle

        self.target = target
        self.direction = int(direction)
        self.place_action = int(place_action)
        dx, dy = _DIR_OFFSETS[self.direction]
        self.stand_tile = (target[0] - dx, target[1] - dy)
        self._navigator = NavigateTo(self.stand_tile)
        self._fired = False

    def step(self, view: WorldView) -> StepReturn:
        if self._fired:
            return Result.DONE, None

        if view.player.pos != self.stand_tile:
            return self._navigator.step(view)

        if view.player.direction != self.direction:
            return Result.RUNNING, face_action(self.direction)

        self._fired = True
        return Result.RUNNING, self.place_action


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
