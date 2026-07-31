"""Tests for splitter / crossing decode tables in :mod:`factoriax.engine.tables`.

The lookup tables drive the per-tick math in
:func:`~factoriax.engine.machines.run_conveyor_belts`, where miswiring the
direction encoding mixes two item streams without raising anything. These
structural tests catch the table errors at import time instead.
"""

from __future__ import annotations

import pytest

from factoriax.engine.constants import Direction
from factoriax.engine.tables import (
    CROSSING_AXIS_DIRS,
    CROSSING_DIAGONAL,
    SPLITTER_PERP_OUTPUTS,
)

# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


def test_splitter_table_shape() -> None:
    """Five rows (NONE + the four Direction values), two outputs each."""
    assert SPLITTER_PERP_OUTPUTS.shape == (5, 2)


def test_splitter_inactive_row_is_zero() -> None:
    """Index 0 (NONE) must be (0, 0) so unset entities push nothing."""
    assert int(SPLITTER_PERP_OUTPUTS[0, 0]) == 0
    assert int(SPLITTER_PERP_OUTPUTS[0, 1]) == 0


@pytest.mark.parametrize(
    "facing, expected",
    [
        (Direction.UP, {Direction.LEFT, Direction.RIGHT}),
        (Direction.DOWN, {Direction.LEFT, Direction.RIGHT}),
        (Direction.LEFT, {Direction.UP, Direction.DOWN}),
        (Direction.RIGHT, {Direction.UP, Direction.DOWN}),
    ],
)
def test_splitter_outputs_are_perpendicular(facing, expected) -> None:
    """Outputs of a splitter facing ``d`` must be the two directions
    perpendicular to ``d``. This is a set comparison, so the slot order
    does not matter if both perpendiculars are present."""
    row = SPLITTER_PERP_OUTPUTS[int(facing)]
    got = {Direction(int(row[0])), Direction(int(row[1]))}
    assert got == expected


def test_splitter_outputs_never_alias_facing() -> None:
    """The output directions for any facing must not include the facing
    itself. That means that the splitter pushes back into its own
    feeder belt and creates a loop."""
    for d in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
        row = SPLITTER_PERP_OUTPUTS[int(d)]
        assert int(row[0]) != int(d)
        assert int(row[1]) != int(d)


# ---------------------------------------------------------------------------
# Crossing
# ---------------------------------------------------------------------------


def test_crossing_table_shape() -> None:
    """Five rows (NONE + the four packed direction encodings), two axes."""
    assert CROSSING_AXIS_DIRS.shape == (5, 2)


def test_crossing_inactive_row_is_zero() -> None:
    assert int(CROSSING_AXIS_DIRS[0, 0]) == 0
    assert int(CROSSING_AXIS_DIRS[0, 1]) == 0


@pytest.mark.parametrize(
    "encoding, expected_vert, expected_horiz",
    [
        (1, Direction.DOWN, Direction.RIGHT),  # N→S + W→E, diagonal \
        (2, Direction.DOWN, Direction.LEFT),  # N→S + E→W, diagonal /
        (3, Direction.UP, Direction.RIGHT),  # S→N + W→E, diagonal /
        (4, Direction.UP, Direction.LEFT),  # S→N + E→W, diagonal \
    ],
)
def test_crossing_axes_decode(encoding, expected_vert, expected_horiz) -> None:
    """Each packed direction encoding (1..4) decodes to the documented
    (vertical_output, horizontal_output) pair."""
    row = CROSSING_AXIS_DIRS[encoding]
    assert Direction(int(row[0])) is expected_vert
    assert Direction(int(row[1])) is expected_horiz


def test_crossing_axes_are_orthogonal() -> None:
    """Vertical output is always UP or DOWN. Horizontal output is always
    LEFT or RIGHT. The two axes never mix across the axis split."""
    vert_set = {int(Direction.UP), int(Direction.DOWN)}
    horiz_set = {int(Direction.LEFT), int(Direction.RIGHT)}
    for d in range(1, 5):
        row = CROSSING_AXIS_DIRS[d]
        assert int(row[0]) in vert_set
        assert int(row[1]) in horiz_set


def test_crossing_diagonal_matches_input_pair() -> None:
    """The diagonal glyph reflects which two adjacent input sides the
    streams enter from. Rule: pair (N, W) and pair (S, E) live on the
    NW-SE diagonal (``\\``); pair (N, E) and pair (S, W) live on
    NE-SW (``/``)."""
    # Encoding 1: vertical input from N, horizontal input from W.
    # Encoding 4: vertical input from S, horizontal input from E.
    assert CROSSING_DIAGONAL[1] == "\\"
    assert CROSSING_DIAGONAL[4] == "\\"
    # Encoding 2: N input + E input. Encoding 3: S input + W input.
    assert CROSSING_DIAGONAL[2] == "/"
    assert CROSSING_DIAGONAL[3] == "/"
