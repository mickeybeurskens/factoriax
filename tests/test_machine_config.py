"""Tests for the per-machine configuration tables.

Per-machine config has two homes, because the engine and the editor share no
field except the buffer cap. Capacities the simulation reads live in
``factoriax.engine.tables``; the slot layout only the editor draws lives in
``factoriax.playground.editor.slot_display``.

These tests lock both against a golden snapshot taken before the config was
first centralised, so the tables cannot silently drift from the shipped values.
Two entries intentionally differ from the pre-refactor arrays, both confined to
slots beyond a machine's slot count (which nothing reads): the slot-view width
shrank ``8 -> 3`` (the real max), and ``PALLET``'s spurious slots 1-7 (formerly
``STORAGE``) are ``NONE`` now that roles derive from its single-slot entry.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine import tables
from factoriax.engine.constants import BlockType, Machine, SlotRole
from factoriax.playground.editor import slot_display

# --- Golden values: the per-machine config, indexed by Machine value -------
# NONE, MINER, PALLET, CONVEYOR_BELT, ASSEMBLER, ARM, ROCKET, FURNACE,
# SCIENCE_LAB, SPLITTER, CROSSING. (DEF3 made Machine the single tag enum, so
# CONVEYOR_BELT=3 and ASSEMBLER=4 -- they swap vs the old MachineType order.)

_GOLDEN_NUM_SLOTS: tuple[int, ...] = (0, 1, 1, 1, 3, 0, 0, 3, 2, 1, 2)
_GOLDEN_MAX_STACK: tuple[int, ...] = (0, 64, 256, 3, 1000, 1, 0, 1000, 1000, 2, 2)
_GOLDEN_MAX_HEALTH: int = 256

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


# --- Engine-side capacities ------------------------------------------------


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


# --- Editor-side slot layout -----------------------------------------------


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


# --- The two tables against the engine's real storage ----------------------
# The engine stores a machine's contents in ``ent_buf`` plus the columns of
# ``ent_asm_in``. These read the real shapes off a constructed state so the
# editor cannot offer more slots than the engine can hold.


def test_slot_width_matches_engine_buffer_capacity(state_factory) -> None:
    """Editor slot-view width equals the engine's per-entity slot capacity.

    Capacity is one ``ent_buf`` slot plus the columns of ``ent_asm_in``; the
    derived width must equal it so the editor shows exactly the slots the
    engine can store, no more and no fewer.
    """
    state = state_factory(
        world_map=jnp.full((2, 2), int(BlockType.DIRT), dtype=jnp.int32)
    )
    ent_asm_in_width = int(state.ent_asm_in_type.shape[1])
    engine_capacity = 1 + ent_asm_in_width
    assert slot_display.MAX_MACHINE_INVENTORY_SLOTS == engine_capacity


def test_no_machine_exceeds_engine_capacity(state_factory) -> None:
    """No machine declares more slots than the engine can hold.

    Total slots fit in ``ent_buf`` plus ``ent_asm_in``; INPUT slots
    specifically live in ``ent_asm_in``, so a machine cannot declare more
    inputs than it has columns.
    """
    state = state_factory(
        world_map=jnp.full((2, 2), int(BlockType.DIRT), dtype=jnp.int32)
    )
    ent_asm_in_width = int(state.ent_asm_in_type.shape[1])
    capacity = 1 + ent_asm_in_width
    for machine, roles in slot_display.MACHINE_SLOTS.items():
        assert len(roles) <= capacity, machine.name
        n_input = sum(1 for role in roles if int(role) == int(SlotRole.INPUT))
        assert n_input <= ent_asm_in_width, machine.name


def test_machine_spec_module_is_gone() -> None:
    """The merged ``machine_spec`` module no longer exists."""
    with pytest.raises(ImportError):
        __import__("factoriax.engine.machine_spec")
