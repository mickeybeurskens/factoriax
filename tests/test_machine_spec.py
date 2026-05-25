"""Tests for the machine specification (DEF1).

S0 (characterization lock): snapshot the exact current contents of the
``MachineType``-indexed config arrays and the machine-config scalars
*before* the DEF1 ``MachineSpec`` refactor derives them from per-machine
records. Any unintended change to these values fails here.

The refactor will make exactly two intentional changes to these golden
values, both confined to slots beyond a machine's ``num_slots`` (which
nothing reads):

1. the editor slot-view width shrinks ``8 -> 3`` (``max`` real slot count),
   so ``MACHINE_SLOT_ROLES`` becomes shape ``(11, 3)`` and
   ``MAX_MACHINE_INVENTORY_SLOTS`` becomes ``3``;
2. ``PALLET``'s spurious slots 1-7 (currently ``STORAGE``) become ``NONE``
   once roles are derived from its single-slot spec — the present
   ``[STORAGE] * 8`` row disagrees with ``MACHINE_NUM_SLOTS[PALLET] == 1``.

Until then this file asserts the current, pre-refactor truth.
"""

from __future__ import annotations

import numpy as np

from factoriax.constants import (
    MACHINE_MAX_STACK,
    MACHINE_MAX_TYPES,
    MACHINE_NUM_SLOTS,
    MACHINE_SLOT_ROLES,
    MAX_HEALTH,
    MAX_MACHINE_INVENTORY_SLOTS,
    MachineType,
)

# --- Golden snapshot of the current per-MachineType config -----------------
# Indexed by MachineType value: NONE, MINER, PALLET, ASSEMBLER, CONVEYOR_BELT,
# ARM, ROCKET, FURNACE, SCIENCE_LAB, SPLITTER, CROSSING.

_GOLDEN_NUM_SLOTS: tuple[int, ...] = (0, 1, 1, 3, 1, 0, 0, 3, 2, 1, 2)
_GOLDEN_MAX_STACK: tuple[int, ...] = (0, 64, 256, 1000, 3, 1, 0, 1000, 1000, 2, 2)
_GOLDEN_MAX_TYPES: tuple[int, ...] = (0, 2, 1, 4, 1, 1, 0, 2, 2, 1, 2)
_GOLDEN_MAX_HEALTH: int = 256
_GOLDEN_MAX_INVENTORY_SLOTS: int = 8

# Per-machine slot roles, width 8 (SlotRole values: NONE=0, INPUT=1, OUTPUT=2,
# STORAGE=3, FUEL=4). PALLET's [3]*8 is the documented drift vs num_slots=1.
_GOLDEN_SLOT_ROLES: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),  # NONE
    (2, 0, 0, 0, 0, 0, 0, 0),  # MINER       — 1 OUTPUT
    (3, 3, 3, 3, 3, 3, 3, 3),  # PALLET      — drift: 8 STORAGE vs num_slots=1
    (1, 1, 2, 0, 0, 0, 0, 0),  # ASSEMBLER   — 2 INPUT, 1 OUTPUT
    (3, 0, 0, 0, 0, 0, 0, 0),  # CONVEYOR_BELT — 1 STORAGE
    (0, 0, 0, 0, 0, 0, 0, 0),  # ARM         — 0 slots
    (0, 0, 0, 0, 0, 0, 0, 0),  # ROCKET      — 0 slots
    (1, 1, 2, 0, 0, 0, 0, 0),  # FURNACE     — 2 INPUT, 1 OUTPUT
    (1, 1, 0, 0, 0, 0, 0, 0),  # SCIENCE_LAB — 2 INPUT
    (3, 0, 0, 0, 0, 0, 0, 0),  # SPLITTER    — 1 STORAGE
    (3, 3, 0, 0, 0, 0, 0, 0),  # CROSSING    — 2 STORAGE
)


def test_num_slots_snapshot() -> None:
    """``MACHINE_NUM_SLOTS`` matches the locked golden values."""
    assert tuple(np.asarray(MACHINE_NUM_SLOTS).tolist()) == _GOLDEN_NUM_SLOTS


def test_max_stack_snapshot() -> None:
    """``MACHINE_MAX_STACK`` matches the locked golden values."""
    assert tuple(np.asarray(MACHINE_MAX_STACK).tolist()) == _GOLDEN_MAX_STACK


def test_max_types_snapshot() -> None:
    """``MACHINE_MAX_TYPES`` matches the locked golden values."""
    assert tuple(np.asarray(MACHINE_MAX_TYPES).tolist()) == _GOLDEN_MAX_TYPES


def test_slot_roles_snapshot() -> None:
    """``MACHINE_SLOT_ROLES`` matches the locked golden rows (width 8)."""
    rows = tuple(tuple(row) for row in np.asarray(MACHINE_SLOT_ROLES).tolist())
    assert rows == _GOLDEN_SLOT_ROLES


def test_scalars_snapshot() -> None:
    """The machine-config scalars match the locked golden values."""
    assert int(MAX_HEALTH) == _GOLDEN_MAX_HEALTH
    assert int(MAX_MACHINE_INVENTORY_SLOTS) == _GOLDEN_MAX_INVENTORY_SLOTS


def test_arrays_are_machinetype_length() -> None:
    """Every per-machine array has one row per ``MachineType`` member."""
    n = len(MachineType)
    assert len(_GOLDEN_NUM_SLOTS) == n
    assert np.asarray(MACHINE_NUM_SLOTS).shape == (n,)
    assert np.asarray(MACHINE_MAX_STACK).shape == (n,)
    assert np.asarray(MACHINE_MAX_TYPES).shape == (n,)
    assert np.asarray(MACHINE_SLOT_ROLES).shape[0] == n
