"""Tests for :mod:`factoriax.engine.tables`.

Three groups live here.

The golden group pins the per-machine capacity tables against hand-written
values, so a change to one has to be deliberate.

The decode group covers the splitter and crossing lookup tables. They drive
the per-tick math in ``run_conveyor_belts``, where a miswired row sends items
the wrong way with nothing to signal it.

The last group asserts that ``SOLID_BLOCKS`` is the single source of what a
player can walk through.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine import tables
from factoriax.engine.constants import BlockType, Direction, Machine
from factoriax.engine.placement import is_valid_placement_tile
from factoriax.engine.step import is_position_walkable
from factoriax.engine.tables import (
    CROSSING_AXIS_DIRS,
    CROSSING_DIAGONAL,
    SOLID_BLOCKS,
    SPLITTER_PERP_OUTPUTS,
)

#: Hand-written capacities. A change to ``tables`` must be deliberate enough to
#: come here as well.
_GOLDEN_MAX_STACK: tuple[int, ...] = (0, 64, 256, 3, 1000, 1, 0, 1000, 1000, 2, 2)
_GOLDEN_MAX_HEALTH: int = 256


def test_max_stack_matches_golden() -> None:
    """``MACHINE_MAX_STACK`` is byte-identical to the golden values."""
    got = tuple(np.asarray(tables.MACHINE_MAX_STACK).tolist())
    assert got == _GOLDEN_MAX_STACK


def test_max_health_is_uniform_across_machines() -> None:
    """``MACHINE_MAX_HEALTH`` spreads the scalar over the Machine range."""
    arr = np.asarray(tables.MACHINE_MAX_HEALTH)
    assert arr.shape == (len(Machine),)
    assert tables.MACHINE_HEALTH == _GOLDEN_MAX_HEALTH
    assert bool((arr == _GOLDEN_MAX_HEALTH).all())


def test_capacity_dtypes_match_the_state_arrays_they_gate() -> None:
    """Both capacity arrays are int16, the dtype of the fields they cap."""
    assert tables.MACHINE_MAX_STACK.dtype == jnp.int16
    assert tables.MACHINE_MAX_HEALTH.dtype == jnp.int16


def test_every_machine_has_a_buffer_entry() -> None:
    """The buffer table covers Machine exactly, with no negative capacity."""
    assert set(tables._MACHINE_BUFFER_STACK) == set(Machine)
    assert min(tables._MACHINE_BUFFER_STACK.values()) >= 0


def test_machine_spec_module_is_gone() -> None:
    """The merged ``machine_spec`` module no longer exists."""
    with pytest.raises(ImportError):
        __import__("factoriax.engine.machine_spec")


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


class TestSolidBlocksIsTheSingleSource:
    """Movement and placement agree on which terrain is solid.

    ``is_position_walkable`` and ``is_valid_placement_tile`` each reject
    solid terrain. Both must read the same set, or a tile becomes walkable
    but unbuildable (or the reverse) with nothing to flag the split.
    """

    @pytest.mark.parametrize("block", list(BlockType))
    def test_walkability_matches_solid_blocks(self, block, state_factory) -> None:
        """Empty terrain is walkable exactly when it is not in SOLID_BLOCKS."""
        state = state_factory(world_map=jnp.array([[block]], dtype=jnp.int8))
        walkable = bool(is_position_walkable(state, jnp.array([0, 0])))
        solid = int(block) in [int(b) for b in SOLID_BLOCKS.tolist()]
        assert walkable is not solid, (
            f"{BlockType(block).name}: walkable={walkable}, in SOLID_BLOCKS={solid}"
        )

    @pytest.mark.parametrize("block", list(BlockType))
    def test_placement_matches_solid_blocks(self, block, state_factory) -> None:
        """Empty terrain is buildable exactly when it is not in SOLID_BLOCKS."""
        state = state_factory(world_map=jnp.array([[block]], dtype=jnp.int8))
        buildable = bool(
            is_valid_placement_tile(state, jnp.array(0), jnp.array(0)),
        )
        solid = int(block) in [int(b) for b in SOLID_BLOCKS.tolist()]
        assert buildable is not solid, (
            f"{BlockType(block).name}: buildable={buildable}, in SOLID_BLOCKS={solid}"
        )

    def test_out_of_bounds_is_not_walkable(self, state_factory) -> None:
        """A query off the map is solid, whatever the stored tile says."""
        state = state_factory(world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int8))
        assert not bool(is_position_walkable(state, jnp.array([5, 5])))
