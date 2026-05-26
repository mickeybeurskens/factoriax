"""Tests for :mod:`baselines.rocket.scripted.layout`.

Three groups of tests:

- :class:`TestExpectedLayoutFromGoals` — the goal-list walker pulls
  out exactly the :class:`PlaceMachineAt` instances and ignores
  everything else.
- :class:`TestDiffLayout` — each of the four
  :data:`~baselines.rocket.scripted.layout.MismatchKind` categories
  (MISSING / WRONG_TYPE / WRONG_DIR / STRAY) is detected
  independently, and a clean layout produces an empty diff.
- :class:`TestVerifyLayout` — the boolean shorthand mirrors
  ``diff_layout`` emptiness.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from baselines.rocket.scripted.goals import (
    MineOre,
    PlaceMachineAt,
    Wait,
)
from baselines.rocket.scripted.layout import (
    diff_layout,
    expected_layout_from_goals,
    verify_layout,
)
from baselines.rocket.scripted.world_model import PlayerScalars, WorldView
from factoriax.constants import Direction, ItemType, Machine


def _make_view(
    machine_type_arr: np.ndarray,
    machine_direction_arr: np.ndarray,
) -> WorldView:
    """Build a :class:`WorldView` carrying only the two arrays the
    layout helpers read; everything else is zero."""
    height, width = machine_type_arr.shape
    zero = np.zeros((height, width), dtype=np.int32)
    bool_zero = np.zeros((height, width), dtype=bool)
    return WorldView(
        block_type=zero,
        machine_type=machine_type_arr,
        block_resources=zero,
        slot0_type=zero,
        slot0_count=zero,
        slot1_type=zero,
        slot1_count=zero,
        slot2_type=zero,
        slot2_count=zero,
        machine_direction=machine_direction_arr,
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
        walkable=bool_zero,
    )


@pytest.fixture
def empty_view() -> WorldView:
    """3x3 grid with no machines placed anywhere."""
    return _make_view(
        machine_type_arr=np.zeros((3, 3), dtype=np.int32),
        machine_direction_arr=np.zeros((3, 3), dtype=np.int32),
    )


@pytest.fixture
def two_pallets_view() -> WorldView:
    """3x3 grid with PALLETs at (0, 0) DOWN and (2, 2) UP."""
    machine_type = np.zeros((3, 3), dtype=np.int32)
    machine_direction = np.zeros((3, 3), dtype=np.int32)
    machine_type[0, 0] = int(Machine.PALLET)
    machine_direction[0, 0] = int(Direction.DOWN)
    machine_type[2, 2] = int(Machine.PALLET)
    machine_direction[2, 2] = int(Direction.UP)
    return _make_view(machine_type, machine_direction)


class TestExpectedLayoutFromGoals:
    """The walker filters PlaceMachineAt and ignores everything else."""

    def test_empty_goal_list(self) -> None:
        assert expected_layout_from_goals([]) == {}

    def test_extracts_single_placement(self) -> None:
        goals = [PlaceMachineAt(Machine.PALLET, (1, 2), int(Direction.DOWN))]
        layout = expected_layout_from_goals(goals)
        assert layout == {(1, 2): (int(Machine.PALLET), int(Direction.DOWN))}

    def test_ignores_non_placement_goals(self) -> None:
        """MineOre, Wait, etc. are passed through but contribute nothing."""
        goals = [
            MineOre(ItemType.IRON_ORE, 5),
            Wait(10),
            PlaceMachineAt(Machine.MINER, (3, 4), int(Direction.UP)),
            Wait(20),
        ]
        layout = expected_layout_from_goals(goals)
        assert layout == {(3, 4): (int(Machine.MINER), int(Direction.UP))}

    def test_duplicate_targets_last_writer_wins(self) -> None:
        """Two placements at the same tile keep the second one."""
        goals = [
            PlaceMachineAt(Machine.PALLET, (1, 1), int(Direction.DOWN)),
            PlaceMachineAt(Machine.MINER, (1, 1), int(Direction.UP)),
        ]
        layout = expected_layout_from_goals(goals)
        assert layout == {(1, 1): (int(Machine.MINER), int(Direction.UP))}


class TestDiffLayout:
    """Each mismatch category is detected; a clean layout has no diff."""

    def test_clean_layout_no_mismatches(self, two_pallets_view: WorldView) -> None:
        expected = {
            (0, 0): (int(Machine.PALLET), int(Direction.DOWN)),
            (2, 2): (int(Machine.PALLET), int(Direction.UP)),
        }
        assert diff_layout(two_pallets_view, expected) == []

    def test_missing_tile(self, empty_view: WorldView) -> None:
        expected = {(1, 1): (int(Machine.PALLET), int(Direction.DOWN))}
        diff = diff_layout(empty_view, expected)
        assert len(diff) == 1
        assert diff[0].kind == "MISSING"
        assert diff[0].tile == (1, 1)
        assert diff[0].observed is None

    def test_wrong_type(self, two_pallets_view: WorldView) -> None:
        expected = {
            (0, 0): (int(Machine.MINER), int(Direction.DOWN)),
            (2, 2): (int(Machine.PALLET), int(Direction.UP)),
        }
        diff = diff_layout(two_pallets_view, expected)
        assert len(diff) == 1
        assert diff[0].kind == "WRONG_TYPE"
        assert diff[0].tile == (0, 0)
        assert diff[0].expected == (
            int(Machine.MINER),
            int(Direction.DOWN),
        )
        assert diff[0].observed == (
            int(Machine.PALLET),
            int(Direction.DOWN),
        )

    def test_wrong_direction(self, two_pallets_view: WorldView) -> None:
        expected = {
            (0, 0): (int(Machine.PALLET), int(Direction.UP)),
            (2, 2): (int(Machine.PALLET), int(Direction.UP)),
        }
        diff = diff_layout(two_pallets_view, expected)
        assert len(diff) == 1
        assert diff[0].kind == "WRONG_DIR"
        assert diff[0].tile == (0, 0)

    def test_stray(self, two_pallets_view: WorldView) -> None:
        """A pallet present at (2, 2) but not in expected is STRAY."""
        expected = {(0, 0): (int(Machine.PALLET), int(Direction.DOWN))}
        diff = diff_layout(two_pallets_view, expected)
        assert len(diff) == 1
        assert diff[0].kind == "STRAY"
        assert diff[0].tile == (2, 2)
        assert diff[0].expected is None

    def test_multiple_mismatches_returned_in_coordinate_order(
        self, empty_view: WorldView
    ) -> None:
        expected = {
            (2, 2): (int(Machine.PALLET), int(Direction.UP)),
            (0, 0): (int(Machine.PALLET), int(Direction.DOWN)),
        }
        diff = diff_layout(empty_view, expected)
        # Both missing; sort returns (0, 0) before (2, 2).
        assert [m.tile for m in diff] == [(0, 0), (2, 2)]

    def test_mismatch_render_human_readable(self, empty_view: WorldView) -> None:
        expected = {(1, 1): (int(Machine.PALLET), int(Direction.DOWN))}
        diff = diff_layout(empty_view, expected)
        line = diff[0].render()
        assert "PALLET" in line
        assert "DOWN" in line
        assert "MISSING" in line


class TestVerifyLayout:
    """Boolean shorthand mirrors emptiness of ``diff_layout``."""

    def test_clean_returns_true(self, two_pallets_view: WorldView) -> None:
        expected = {
            (0, 0): (int(Machine.PALLET), int(Direction.DOWN)),
            (2, 2): (int(Machine.PALLET), int(Direction.UP)),
        }
        assert verify_layout(two_pallets_view, expected) is True

    def test_any_mismatch_returns_false(self, two_pallets_view: WorldView) -> None:
        # Wrong direction at (0, 0) is enough to flip the result.
        expected = {
            (0, 0): (int(Machine.PALLET), int(Direction.UP)),
            (2, 2): (int(Machine.PALLET), int(Direction.UP)),
        }
        assert verify_layout(two_pallets_view, expected) is False

    def test_empty_view_with_empty_plan_returns_true(
        self, empty_view: WorldView
    ) -> None:
        assert verify_layout(empty_view, {}) is True


class TestStrayDetectionRespectsExpected:
    """When a tile is *both* expected and present, it isn't STRAY."""

    def test_expected_present_tile_not_stray(self, two_pallets_view: WorldView) -> None:
        # Plan only mentions (0, 0); (2, 2) is unmentioned and present.
        expected = {(0, 0): (int(Machine.PALLET), int(Direction.DOWN))}
        diff = diff_layout(two_pallets_view, expected)
        assert len(diff) == 1
        assert diff[0].kind == "STRAY"
        assert diff[0].tile == (2, 2)

    def test_partial_plan_missing_plus_stray(self, two_pallets_view: WorldView) -> None:
        """Plan asks for (1, 1) (missing), env has (0, 0) and (2, 2)
        (both stray)."""
        expected = {(1, 1): (int(Machine.PALLET), int(Direction.DOWN))}
        diff = diff_layout(two_pallets_view, expected)
        kinds = [m.kind for m in diff]
        assert kinds.count("MISSING") == 1
        assert kinds.count("STRAY") == 2


class TestNumpyArrayCoordinateConvention:
    """Smoke test that array indexing matches the (x, y) -> [y, x] convention."""

    def test_x_y_targets_correct_array_cell(self) -> None:
        """Place a PALLET at array cell [3, 5] and verify that
        ``expected = {(5, 3): (PALLET, DOWN)}`` matches.

        This pins the convention: ``tile = (x, y)`` indexes
        ``array[y, x]``. A regression here would fail every layout
        diff in the world.
        """
        machine_type = np.zeros((10, 10), dtype=np.int32)
        machine_direction = np.zeros((10, 10), dtype=np.int32)
        machine_type[3, 5] = int(Machine.PALLET)
        machine_direction[3, 5] = int(Direction.DOWN)
        view = _make_view(machine_type, machine_direction)
        expected = {(5, 3): (int(Machine.PALLET), int(Direction.DOWN))}
        diff = diff_layout(view, expected)
        assert diff == [], f"unexpected mismatches: {diff}"


class TestRenderInDifferentKinds:
    """``render`` should produce a single-line string for every kind."""

    @pytest.mark.parametrize(
        "kind,expected_pair,observed_pair",
        [
            (
                "MISSING",
                (int(Machine.PALLET), int(Direction.DOWN)),
                None,
            ),
            (
                "STRAY",
                None,
                (int(Machine.MINER), int(Direction.UP)),
            ),
        ],
    )
    def test_render_handles_none_sides(
        self,
        kind: str,
        expected_pair: tuple[int, int] | None,
        observed_pair: tuple[int, int] | None,
    ) -> None:
        from baselines.rocket.scripted.layout import LayoutMismatch

        m = LayoutMismatch(
            tile=(1, 2),
            expected=expected_pair,
            observed=observed_pair,
            kind=kind,  # type: ignore[arg-type]
        )
        line = m.render()
        assert kind in line
        # ``render`` right-aligns coords to width 2, so (1, 2) prints
        # as "( 1,  2)" — just check the integers are present.
        assert "1" in line and "2" in line

    def test_replace_dataclass_immutable(self) -> None:
        """LayoutMismatch is frozen — replace is the only way to copy."""
        from baselines.rocket.scripted.layout import LayoutMismatch

        m = LayoutMismatch(tile=(0, 0), expected=None, observed=None, kind="STRAY")
        with pytest.raises(Exception):
            m.tile = (9, 9)  # type: ignore[misc]
        # ``replace`` works.
        m2 = replace(m, tile=(1, 1))
        assert m2.tile == (1, 1)
