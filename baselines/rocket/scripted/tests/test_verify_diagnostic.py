"""Tests for the verify-failure diagnostic pipeline.

Three concerns:

- :meth:`VerifyDiagnostic.format` produces a multi-line string that
  includes the tick, goal repr, and any per-goal details.
- The planner wires :meth:`Goal.verify_failure_details` into the
  diagnostic so subclass-specific information is visible to callers.
- :class:`VerifyLayout` truncates large mismatch lists at the
  documented cap and signals truncation with a ``(... N more)``
  trailer.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.rocket.scripted.goals import (
    _MAX_LAYOUT_MISMATCHES_RENDERED,
    PlaceMachineAt,
    VerifyLayout,
)
from baselines.rocket.scripted.planner import Planner, VerifyDiagnostic
from baselines.rocket.scripted.skills import Result
from baselines.rocket.scripted.world_model import PlayerScalars, WorldView
from factoriax.engine.constants import Direction, Machine


def _empty_view(size: int = 32) -> WorldView:
    """A blank ``size x size`` :class:`WorldView` with no machines."""
    zero = np.zeros((size, size), dtype=np.int32)
    return WorldView(
        block_type=zero,
        machine_type=zero,
        block_resources=zero,
        slot0_type=zero,
        slot0_count=zero,
        slot1_type=zero,
        slot1_count=zero,
        slot2_type=zero,
        slot2_count=zero,
        machine_direction=zero,
        buffer_type=zero,
        player=PlayerScalars(
            pos_x=0,
            pos_y=0,
            direction=int(Direction.DOWN),
            timestep=0,
            facing_machine_type=0,
            facing_buffer_type=0,
            facing_buffer_count=0,
            inventory=np.zeros(64, dtype=np.int32),
        ),
        walkable=np.zeros((size, size), dtype=bool),
    )


class TestVerifyDiagnosticFormat:
    """Multi-line render of a :class:`VerifyDiagnostic`."""

    def test_format_without_details(self) -> None:
        diag = VerifyDiagnostic(
            tick=42,
            goal_name="StubGoal",
            goal_repr="StubGoal()",
            action_taken="halt",
        )
        text = diag.format()
        assert "tick=42" in text
        assert "halt" in text
        assert "StubGoal()" in text

    def test_format_with_details(self) -> None:
        diag = VerifyDiagnostic(
            tick=7,
            goal_name="X",
            goal_repr="X()",
            action_taken="halt",
            details="  expected: A\n  observed: B",
        )
        text = diag.format()
        # Both head and detail lines present.
        assert "tick=7" in text
        assert "expected: A" in text
        assert "observed: B" in text


class TestPlaceMachineAtDetails:
    """``PlaceMachineAt`` reports its expected vs observed pair."""

    def test_details_reference_target_tile(self) -> None:
        view = _empty_view()
        goal = PlaceMachineAt(Machine.PALLET, (5, 6), int(Direction.DOWN))
        text = goal.verify_failure_details(view)
        assert text is not None
        assert "(5, 6)" in text
        assert "PALLET" in text
        assert "DOWN" in text


class TestPlannerThreadsDetailsThrough:
    """Planner pulls details from the goal and stuffs them in the diagnostic."""

    def test_planner_diagnostic_carries_details(self) -> None:
        view = _empty_view(size=8)
        # Plan asks for a pallet at (3, 3); empty world ensures verify fails.
        goal = PlaceMachineAt(Machine.PALLET, (3, 3), int(Direction.DOWN))
        # Force the goal to immediately report DONE so verify runs on the
        # first tick — bypass step's "machine count grew?" gate by
        # poking the start-count to 0.
        goal._start_count = -1  # any negative number triggers DONE next tick.

        planner = Planner([goal], debug_log=True)
        # First tick — step's "view.total_machines() > _start_count" gate
        # is satisfied (0 > -1) so step returns DONE; verify then runs.
        result, _ = planner.step(view)

        assert result is Result.FAIL  # halt action.
        diag = planner.verify_diagnostic
        assert diag is not None
        assert diag.details is not None
        assert "PALLET" in diag.details


class TestVerifyLayoutTruncation:
    """The 20-line cap on :class:`VerifyLayout`'s rendered mismatches."""

    def test_no_truncation_for_small_diff(self) -> None:
        view = _empty_view()
        expected = {
            (i, 0): (int(Machine.PALLET), int(Direction.DOWN)) for i in range(3)
        }
        goal = VerifyLayout(expected, label="small")
        text = goal.verify_failure_details(view)
        assert text is not None
        assert "(... " not in text  # no truncation trailer.

    def test_cap_at_documented_constant(self) -> None:
        """When the diff exceeds the cap, only the cap is rendered + a trailer."""
        view = _empty_view()
        n = _MAX_LAYOUT_MISMATCHES_RENDERED + 7
        expected = {
            (i, 0): (int(Machine.PALLET), int(Direction.DOWN)) for i in range(n)
        }
        goal = VerifyLayout(expected, label="big")
        text = goal.verify_failure_details(view)
        assert text is not None
        assert "(... 7 more)" in text
        # Count only the mismatch lines (those beginning with two
        # spaces and a paren); skip the "label:" line.
        line_count = sum(
            1
            for line in text.splitlines()
            if line.startswith("  (") and "MISSING" in line
        )
        assert line_count == _MAX_LAYOUT_MISMATCHES_RENDERED

    def test_passing_layout_returns_no_details(self) -> None:
        """A clean layout produces no detail string (verify returned True)."""
        view = _empty_view()
        # Empty plan == clean diff.
        goal = VerifyLayout({}, label="empty")
        assert goal.verify_failure_details(view) is None


@pytest.mark.parametrize("action", ["halt", "ignore"])
def test_planner_format_round_trip(action: str) -> None:
    """``planner.verify_diagnostic.format()`` runs without error and
    produces a non-empty string for every action taken."""
    view = _empty_view()
    expected = {(2, 2): (int(Machine.PALLET), int(Direction.DOWN))}

    # Subclass per-action so ``verify_failure_action`` is set as a
    # proper class variable rather than mutated on the instance.
    class _Subclass(VerifyLayout):
        verify_failure_action = action  # type: ignore[assignment]

    goal = _Subclass(expected, label=f"{action}-test")
    planner = Planner([goal], debug_log=True)
    planner.step(view)
    diag = planner.verify_diagnostic
    assert diag is not None
    text = diag.format()
    assert text
    assert "PALLET" in text
