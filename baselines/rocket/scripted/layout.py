"""Layout snapshotting + diff for the scripted rocket agent.

Two halves:

- :func:`expected_layout_from_goals` walks a goal list and pulls out
  every :class:`~baselines.rocket.scripted.goals.PlaceMachineAt`
  instance, projecting it to a ``(x, y) -> (machine_type, direction)``
  dict. Composite helpers like
  :func:`~baselines.rocket.scripted.goals.build_smelter_cell_at`
  already unroll into flat ``PlaceMachineAt`` lists, so the walker is
  a one-pass filter.
- :func:`diff_layout` compares an expected dict against the live
  :class:`~baselines.rocket.scripted.world_model.WorldView` arrays
  and returns a list of :class:`LayoutMismatch` records (one per
  divergent tile). :func:`verify_layout` is the boolean shorthand.

Why a dedicated module: layout diffing is reused by the
:class:`~baselines.rocket.scripted.goals.VerifyLayout` stage gate
*and* by post-hoc diagnostic tooling (compare-agents reports,
per-stage assertions in tests). Keeping the helpers in one place
means callers don't reach into goal internals.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from factoriax.constants import Direction, MachineType

from .goals import Goal, PlaceMachineAt, PlaceMachineFromBackAt
from .world_model import WorldView

#: Type alias for an expected layout. Maps ``(x, y)`` tile coords to
#: the ``(machine_type, direction)`` pair that's supposed to land
#: there. Both fields are stored as plain ints to keep the dict cheap
#: to construct, hash, and serialise.
ExpectedLayout = dict[tuple[int, int], tuple[int, int]]

#: Categories of layout divergence. Each maps to one
#: :class:`LayoutMismatch.kind` value.
MismatchKind = Literal["MISSING", "WRONG_TYPE", "WRONG_DIR", "STRAY"]


@dataclass(frozen=True)
class LayoutMismatch:
    """One tile where observed reality diverges from the plan.

    Attributes:
        tile: ``(x, y)`` coordinate of the divergent tile.
        expected: ``(machine_type, direction)`` the plan called for,
            or ``None`` when the kind is ``"STRAY"`` (something
            placed that wasn't planned).
        observed: ``(machine_type, direction)`` actually present, or
            ``None`` when the kind is ``"MISSING"`` (the tile is
            empty).
        kind: Which category of divergence applies — see
            :data:`MismatchKind`.
    """

    tile: tuple[int, int]
    expected: tuple[int, int] | None
    observed: tuple[int, int] | None
    kind: MismatchKind

    def render(self) -> str:
        """Format this mismatch as a single human-readable line.

        Both ``mt == NONE`` (0) and ``direction == 0`` are the zero-init
        values for the underlying arrays — neither is a valid
        :class:`Direction` (the enum starts at 1) so we render them as
        ``NONE`` rather than tripping
        :meth:`enum.Enum.__call__` with a ``ValueError``.
        """
        x, y = self.tile

        def _dir_name(d: int) -> str:
            return Direction(d).name if d != 0 else "NONE"

        def _fmt(pair: tuple[int, int] | None) -> str:
            if pair is None:
                return "NONE"
            mt, d = pair
            return f"({MachineType(mt).name}, {_dir_name(d)})"

        return (
            f"  ({x:>2}, {y:>2}) {self.kind:<10}  "
            f"expected={_fmt(self.expected)}  observed={_fmt(self.observed)}"
        )


def expected_layout_from_goals(goals: Iterable[Goal]) -> ExpectedLayout:
    """Project a goal list to its expected machine layout.

    Walks ``goals`` and pulls out every
    :class:`~baselines.rocket.scripted.goals.PlaceMachineAt`,
    indexing by ``goal.target`` and storing
    ``(goal.machine_type, goal.facing)`` per tile. A later goal that
    targets the same tile overwrites the earlier entry — the agent
    only ever lands one machine per tile, so a duplicate target is
    almost certainly a bug, but the dict semantics give the
    last-writer the canonical entry for diff purposes.

    Composite helpers in :mod:`~baselines.rocket.scripted.goals`
    (``build_smelter_cell_at``, ``place_belt_network``,
    ``place_ore_node``) return flat lists of ``PlaceMachineAt``
    instances at construction time, so this walker doesn't need to
    recurse — every expected placement is already visible at the top
    level of the goal list.

    Args:
        goals: Iterable of :class:`Goal` (typically the planner's
            queue or any sub-slice of it).

    Returns:
        Dict mapping ``(x, y)`` to ``(machine_type, direction)``.
    """
    layout: ExpectedLayout = {}
    for goal in goals:
        if isinstance(goal, (PlaceMachineAt, PlaceMachineFromBackAt)):
            layout[goal.target] = (goal.machine_type, goal.facing)
    return layout


def diff_layout(view: WorldView, expected: ExpectedLayout) -> list[LayoutMismatch]:
    """Diff observed machine state against an expected layout.

    Three observed-vs-expected categories arise:

    - tile in *expected* and ``view.machine_type[y, x] == NONE``: the
      placement never landed (``MISSING``).
    - tile in *expected* and observed type/direction differ:
      ``WRONG_TYPE`` or ``WRONG_DIR``.
    - tile *not* in expected but observed type is non-NONE: the
      agent placed something we didn't ask for (``STRAY``). Sanity
      check; this should never fire for a clean plan.

    Pre-placed machines (the rocket scenario's pre-spawned furnace
    + assembler) are not in the expected dict; they show up as
    ``STRAY``. Callers that want to ignore those should pre-populate
    their expected dict with the pre-placed entries.

    Args:
        view: Current :class:`WorldView`.
        expected: Plan-derived layout from
            :func:`expected_layout_from_goals` (or hand-built).

    Returns:
        Empty list when the layout matches. Otherwise one
        :class:`LayoutMismatch` per divergent tile, in coordinate
        order.
    """
    mismatches: list[LayoutMismatch] = []
    none_mt = int(MachineType.NONE)
    seen_expected: set[tuple[int, int]] = set()

    for tile, expected_pair in sorted(expected.items()):
        seen_expected.add(tile)
        x, y = tile
        observed_mt = int(view.machine_type[y, x])
        observed_dir = int(view.machine_direction[y, x])
        if observed_mt == none_mt:
            mismatches.append(
                LayoutMismatch(
                    tile=tile,
                    expected=expected_pair,
                    observed=None,
                    kind="MISSING",
                )
            )
            continue
        if observed_mt != expected_pair[0]:
            mismatches.append(
                LayoutMismatch(
                    tile=tile,
                    expected=expected_pair,
                    observed=(observed_mt, observed_dir),
                    kind="WRONG_TYPE",
                )
            )
            continue
        if observed_dir != expected_pair[1]:
            mismatches.append(
                LayoutMismatch(
                    tile=tile,
                    expected=expected_pair,
                    observed=(observed_mt, observed_dir),
                    kind="WRONG_DIR",
                )
            )

    # Stray detection: every non-NONE tile in observed that wasn't in expected.
    height, width = view.machine_type.shape
    for y in range(height):
        for x in range(width):
            tile = (x, y)
            if tile in seen_expected:
                continue
            observed_mt = int(view.machine_type[y, x])
            if observed_mt == none_mt:
                continue
            observed_dir = int(view.machine_direction[y, x])
            mismatches.append(
                LayoutMismatch(
                    tile=tile,
                    expected=None,
                    observed=(observed_mt, observed_dir),
                    kind="STRAY",
                )
            )

    return mismatches


def verify_layout(view: WorldView, expected: ExpectedLayout) -> bool:
    """Boolean shorthand: ``True`` iff :func:`diff_layout` is empty."""
    return not diff_layout(view, expected)
