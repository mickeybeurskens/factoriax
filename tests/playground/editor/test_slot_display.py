"""Tests for :mod:`factoriax.playground.editor.slot_display`.

The editor draws one slot per thing a machine can hold. ``MACHINE_SLOTS``
declares the role of each slot, and ``MAX_MACHINE_INVENTORY_SLOTS`` is the
width of the widest machine. The golden values below are the pre-refactor
roles, kept so a change to the derived table has to be deliberate.
"""

from __future__ import annotations

import numpy as np

from factoriax.engine.constants import Machine
from factoriax.playground.editor import slot_display

#: Pre-refactor slot count per machine, kept as a golden value.
_GOLDEN_NUM_SLOTS: tuple[int, ...] = (0, 1, 1, 1, 3, 0, 0, 3, 2, 1, 2)

# Pre-refactor slot roles, width 8 (SlotRole: NONE=0, INPUT=1, OUTPUT=2,
# STORAGE=3, FUEL=4). PALLET's [3]*8 was the drift vs num_slots=1; kept here
# so the agreement test can prove the within-num_slots roles are unchanged.
_GOLDEN_SLOT_ROLES: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),  # NONE
    (2, 0, 0, 0, 0, 0, 0, 0),  # MINER         - 1 OUTPUT
    (3, 3, 3, 3, 3, 3, 3, 3),  # PALLET        - drift: 8 STORAGE vs 1 slot
    (3, 0, 0, 0, 0, 0, 0, 0),  # CONVEYOR_BELT - 1 STORAGE
    (1, 1, 2, 0, 0, 0, 0, 0),  # ASSEMBLER     - 2 INPUT, 1 OUTPUT
    (0, 0, 0, 0, 0, 0, 0, 0),  # ARM           - 0 slots
    (0, 0, 0, 0, 0, 0, 0, 0),  # ROCKET        - 0 slots
    (1, 1, 2, 0, 0, 0, 0, 0),  # FURNACE       - 2 INPUT, 1 OUTPUT
    (1, 1, 0, 0, 0, 0, 0, 0),  # SCIENCE_LAB   - 2 INPUT
    (3, 0, 0, 0, 0, 0, 0, 0),  # SPLITTER      - 1 STORAGE
    (3, 3, 0, 0, 0, 0, 0, 0),  # CROSSING      - 2 STORAGE
)


# Derived roles after the refactor: width 3 (real max), PALLET corrected.
_DERIVED_SLOT_ROLES: tuple[tuple[int, ...], ...] = (
    (0, 0, 0),  # NONE
    (2, 0, 0),  # MINER
    (3, 0, 0),  # PALLET       (fixed: one STORAGE slot)
    (3, 0, 0),  # CONVEYOR_BELT
    (1, 1, 2),  # ASSEMBLER
    (0, 0, 0),  # ARM
    (0, 0, 0),  # ROCKET
    (1, 1, 2),  # FURNACE
    (1, 1, 0),  # SCIENCE_LAB
    (3, 0, 0),  # SPLITTER
    (3, 3, 0),  # CROSSING
)


def test_num_slots_matches_golden() -> None:
    """``MACHINE_NUM_SLOTS`` is byte-identical to the golden values."""
    assert tuple(slot_display.MACHINE_NUM_SLOTS.tolist()) == _GOLDEN_NUM_SLOTS


def test_width_is_the_real_max() -> None:
    """The slot-view width derives to the real max (3), not the old 8."""
    assert slot_display.MAX_MACHINE_INVENTORY_SLOTS == 3
    assert np.asarray(slot_display.MACHINE_SLOT_ROLES).shape == (len(Machine), 3)


def test_slot_roles_match_derived_expectation() -> None:
    """Derived roles match the width-3, PALLET-corrected expectation."""
    rows = tuple(
        tuple(row) for row in np.asarray(slot_display.MACHINE_SLOT_ROLES).tolist()
    )
    assert rows == _DERIVED_SLOT_ROLES


def test_roles_agree_with_golden_within_num_slots() -> None:
    """Role changes are confined beyond each machine's slot count.

    For every machine, the derived roles up to its slot count equal the golden
    roles up to that count, proving the only differences (PALLET's slots 1-7,
    the dropped columns) are slots nothing reads.
    """
    derived = np.asarray(slot_display.MACHINE_SLOT_ROLES)
    for i, n in enumerate(_GOLDEN_NUM_SLOTS):
        assert tuple(derived[i, :n].tolist()) == _GOLDEN_SLOT_ROLES[i][:n], (
            f"machine {Machine(i).name} role within num_slots changed"
        )


def test_every_machine_has_a_slot_entry() -> None:
    """The slot table covers Machine exactly."""
    assert set(slot_display.MACHINE_SLOTS) == set(Machine)


def test_num_slots_counts_the_declared_roles() -> None:
    """``MACHINE_NUM_SLOTS[m]`` is the length of ``MACHINE_SLOTS[m]``."""
    for machine, roles in slot_display.MACHINE_SLOTS.items():
        assert int(slot_display.MACHINE_NUM_SLOTS[int(machine)]) == len(roles)
