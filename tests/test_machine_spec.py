"""Tests for the machine specification (DEF1).

``machine_spec`` is the single source of truth for per-machine config:
each machine is one ``MachineSpec`` record, and the ``MachineType``-indexed
arrays the engine and editor index are derived from those records. These
tests lock the derived arrays against the values they replaced — a golden
snapshot taken before the refactor — so the derivation can never silently
drift from the shipped config.

Two values intentionally differ from the pre-refactor arrays, both confined
to slots beyond a machine's ``num_slots`` (which nothing reads): the
slot-view width shrank ``8 -> 3`` (the real max), and ``PALLET``'s spurious
slots 1-7 (formerly ``STORAGE``) are ``NONE`` now that roles derive from its
single-slot spec.
"""

from __future__ import annotations

import numpy as np

from factoriax import machine_spec
from factoriax.constants import MachineType

# --- Golden values: the per-machine config as it shipped pre-refactor ------
# Indexed by MachineType value: NONE, MINER, PALLET, ASSEMBLER, CONVEYOR_BELT,
# ARM, ROCKET, FURNACE, SCIENCE_LAB, SPLITTER, CROSSING.

_GOLDEN_NUM_SLOTS: tuple[int, ...] = (0, 1, 1, 3, 1, 0, 0, 3, 2, 1, 2)
_GOLDEN_MAX_STACK: tuple[int, ...] = (0, 64, 256, 1000, 3, 1, 0, 1000, 1000, 2, 2)
_GOLDEN_MAX_TYPES: tuple[int, ...] = (0, 2, 1, 4, 1, 1, 0, 2, 2, 1, 2)
_GOLDEN_MAX_HEALTH: int = 256

# Pre-refactor slot roles, width 8 (SlotRole: NONE=0, INPUT=1, OUTPUT=2,
# STORAGE=3, FUEL=4). PALLET's [3]*8 was the drift vs num_slots=1; kept here
# so the agreement test can prove the within-num_slots roles are unchanged.
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

# Derived roles after the refactor: width 3 (real max), PALLET corrected.
_DERIVED_SLOT_ROLES: tuple[tuple[int, ...], ...] = (
    (0, 0, 0),  # NONE
    (2, 0, 0),  # MINER
    (3, 0, 0),  # PALLET       (fixed: one STORAGE slot)
    (1, 1, 2),  # ASSEMBLER
    (3, 0, 0),  # CONVEYOR_BELT
    (0, 0, 0),  # ARM
    (0, 0, 0),  # ROCKET
    (1, 1, 2),  # FURNACE
    (1, 1, 0),  # SCIENCE_LAB
    (3, 0, 0),  # SPLITTER
    (3, 3, 0),  # CROSSING
)


def test_derived_num_slots_match_golden() -> None:
    """Derived ``MACHINE_NUM_SLOTS`` is byte-identical to the golden values."""
    assert tuple(machine_spec.MACHINE_NUM_SLOTS.tolist()) == _GOLDEN_NUM_SLOTS


def test_derived_max_stack_match_golden() -> None:
    """Derived ``MACHINE_MAX_STACK`` is byte-identical to the golden values."""
    got = tuple(np.asarray(machine_spec.MACHINE_MAX_STACK).tolist())
    assert got == _GOLDEN_MAX_STACK


def test_derived_max_types_match_golden() -> None:
    """Derived ``MACHINE_MAX_TYPES`` is byte-identical to the golden values."""
    got = tuple(np.asarray(machine_spec.MACHINE_MAX_TYPES).tolist())
    assert got == _GOLDEN_MAX_TYPES


def test_derived_max_health_is_default_for_every_machine() -> None:
    """Derived ``MACHINE_MAX_HEALTH`` is the default health per machine."""
    arr = np.asarray(machine_spec.MACHINE_MAX_HEALTH)
    assert arr.shape == (len(MachineType),)
    assert bool((arr == _GOLDEN_MAX_HEALTH).all())


def test_derived_width_shrinks_to_real_max() -> None:
    """The slot-view width derives to the real max (3), not the old 8."""
    assert machine_spec.MAX_MACHINE_INVENTORY_SLOTS == 3
    assert np.asarray(machine_spec.MACHINE_SLOT_ROLES).shape == (len(MachineType), 3)


def test_derived_slot_roles() -> None:
    """Derived roles match the width-3, PALLET-corrected expectation."""
    rows = tuple(
        tuple(row) for row in np.asarray(machine_spec.MACHINE_SLOT_ROLES).tolist()
    )
    assert rows == _DERIVED_SLOT_ROLES


def test_derived_roles_agree_within_num_slots() -> None:
    """Role changes are confined beyond each machine's num_slots.

    For every machine, the derived roles up to ``num_slots`` equal the
    golden roles up to ``num_slots`` — proving the only differences
    (PALLET's slots 1-7, the dropped columns) are slots nothing reads.
    """
    derived = np.asarray(machine_spec.MACHINE_SLOT_ROLES)
    for i, n in enumerate(_GOLDEN_NUM_SLOTS):
        assert tuple(derived[i, :n].tolist()) == _GOLDEN_SLOT_ROLES[i][:n], (
            f"machine {MachineType(i).name} role within num_slots changed"
        )


def test_arrays_are_machinetype_length() -> None:
    """Golden data and derived arrays have one row per ``MachineType``."""
    n = len(MachineType)
    assert len(_GOLDEN_NUM_SLOTS) == n
    assert np.asarray(machine_spec.MACHINE_NUM_SLOTS).shape == (n,)
    assert np.asarray(machine_spec.MACHINE_SLOT_ROLES).shape[0] == n


def test_specs_cover_machinetypes_in_order() -> None:
    """One spec per MachineType, in value order (the validate invariant)."""
    assert len(machine_spec.MACHINE_SPECS) == len(MachineType)
    for i, spec in enumerate(machine_spec.MACHINE_SPECS):
        assert spec.machine_type == MachineType(i)
