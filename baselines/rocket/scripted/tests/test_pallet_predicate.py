"""Unit tests for ``_pallet_has_output_predicate``.

The predicate gates stage transitions in the staged-bus-pull
architecture. A stage that places a cell must wait for that cell's
output pallet to actually contain the expected item before the next
stage's bus pull is allowed to ``WithdrawFromBusAt`` the same pallet
-- otherwise the withdraw idles out against an empty buffer.

These tests exercise the predicate against synthetic
:class:`WorldView` snapshots without invoking the env.
"""

from __future__ import annotations

import numpy as np

from baselines.rocket.scripted.agent import _pallet_has_output_predicate
from baselines.rocket.scripted.world_model import PlayerScalars, WorldView
from factoriax.constants import Direction, ItemType, MachineType


def _make_view_with_pallet(
    tile: tuple[int, int],
    buf_type: int,
    buf_count: int,
    *,
    map_size: int = 8,
) -> WorldView:
    """Build a minimal :class:`WorldView` with one PALLET tile populated.

    The pallet's slot-2 (output buffer) holds *buf_count* items of
    *buf_type*; every other tile is empty.
    """
    zero = np.zeros((map_size, map_size), dtype=np.int32)
    bool_zero = np.zeros((map_size, map_size), dtype=bool)
    machine_type = np.zeros((map_size, map_size), dtype=np.int32)
    slot2_type = np.zeros((map_size, map_size), dtype=np.int32)
    slot2_count = np.zeros((map_size, map_size), dtype=np.int32)
    buffer_type = np.zeros((map_size, map_size), dtype=np.int32)

    x, y = tile
    machine_type[y, x] = int(MachineType.PALLET)
    slot2_type[y, x] = buf_type
    slot2_count[y, x] = buf_count
    buffer_type[y, x] = buf_type

    return WorldView(
        block_type=zero,
        machine_type=machine_type,
        block_resources=zero,
        slot0_type=zero,
        slot0_count=zero,
        slot1_type=zero,
        slot1_count=zero,
        slot2_type=slot2_type,
        slot2_count=slot2_count,
        machine_direction=zero,
        buffer_type=buffer_type,
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


class TestPalletHasOutputPredicate:
    def test_returns_true_when_pallet_holds_expected_item(self) -> None:
        view = _make_view_with_pallet((3, 4), int(ItemType.WIRE), buf_count=1)
        pred = _pallet_has_output_predicate((3, 4), ItemType.WIRE)
        assert pred(view) is True

    def test_returns_false_when_pallet_is_empty(self) -> None:
        view = _make_view_with_pallet((3, 4), int(ItemType.WIRE), buf_count=0)
        pred = _pallet_has_output_predicate((3, 4), ItemType.WIRE)
        assert pred(view) is False

    def test_returns_false_when_pallet_holds_wrong_item(self) -> None:
        # Pallet holds CIRCUITs but the predicate is gating on FRAME.
        view = _make_view_with_pallet((3, 4), int(ItemType.CIRCUIT), buf_count=5)
        pred = _pallet_has_output_predicate((3, 4), ItemType.FRAME)
        assert pred(view) is False

    def test_returns_false_when_other_tile_holds_item(self) -> None:
        # WIRE is in pallet at (1, 1); predicate gates on (3, 4).
        view = _make_view_with_pallet((1, 1), int(ItemType.WIRE), buf_count=10)
        pred = _pallet_has_output_predicate((3, 4), ItemType.WIRE)
        assert pred(view) is False

    def test_accepts_int_or_itemtype(self) -> None:
        """``item`` accepts both ``ItemType`` and raw int."""
        view = _make_view_with_pallet((2, 2), int(ItemType.MOTOR), buf_count=3)
        pred_enum = _pallet_has_output_predicate((2, 2), ItemType.MOTOR)
        pred_int = _pallet_has_output_predicate((2, 2), int(ItemType.MOTOR))
        assert pred_enum(view) is True
        assert pred_int(view) is True

    def test_predicate_is_pure(self) -> None:
        """Calling the predicate doesn't mutate the view or close over state."""
        view_full = _make_view_with_pallet((0, 0), int(ItemType.WIRE), buf_count=5)
        view_empty = _make_view_with_pallet((0, 0), int(ItemType.WIRE), buf_count=0)
        pred = _pallet_has_output_predicate((0, 0), ItemType.WIRE)
        # Same predicate, different views -> independent answers.
        assert pred(view_full) is True
        assert pred(view_empty) is False
        # And calling again returns the same answer.
        assert pred(view_full) is True
