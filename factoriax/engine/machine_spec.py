"""Per-machine definitions — the single source of truth for machine config.

Each machine type is defined once, completely, as a :class:`MachineSpec`
record. The ``Machine``-indexed arrays the engine and editor index
(``MACHINE_NUM_SLOTS``, ``MACHINE_SLOT_ROLES``, ``MACHINE_MAX_STACK``,
``MACHINE_MAX_TYPES``) are derived from those records and validated at
construction, so a machine's slot count, slot roles, buffer cap, and type
cap can no longer drift apart the way separate hand-maintained arrays did.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import Machine, SlotRole


@dataclass(frozen=True)
class MachineSpec:
    """Complete definition of one machine type."""

    machine_type: Machine
    slots: tuple[SlotRole, ...]
    buffer_stack: int
    max_types: int
    max_health: int

    @property
    def num_slots(self) -> int:
        """Number of logical inventory slots."""
        return len(self.slots)


#: Default maximum health every machine spec is built with.
MAX_HEALTH: int = 256

# One record per Machine, in value order; index == Machine value.
MACHINE_SPECS: tuple[MachineSpec, ...] = (
    MachineSpec(Machine.NONE, (), 0, 0, MAX_HEALTH),
    MachineSpec(Machine.MINER, (SlotRole.OUTPUT,), 64, 2, MAX_HEALTH),
    MachineSpec(Machine.PALLET, (SlotRole.STORAGE,), 256, 1, MAX_HEALTH),
    MachineSpec(Machine.CONVEYOR_BELT, (SlotRole.STORAGE,), 3, 1, MAX_HEALTH),
    MachineSpec(
        Machine.ASSEMBLER,
        (SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT),
        1000,
        4,
        MAX_HEALTH,
    ),
    MachineSpec(Machine.ARM, (), 1, 1, MAX_HEALTH),
    MachineSpec(Machine.ROCKET, (), 0, 0, MAX_HEALTH),
    MachineSpec(
        Machine.FURNACE,
        (SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT),
        1000,
        2,
        MAX_HEALTH,
    ),
    MachineSpec(
        Machine.SCIENCE_LAB,
        (SlotRole.INPUT, SlotRole.INPUT),
        1000,
        2,
        MAX_HEALTH,
    ),
    MachineSpec(Machine.SPLITTER, (SlotRole.STORAGE,), 2, 1, MAX_HEALTH),
    MachineSpec(
        Machine.CROSSING,
        (SlotRole.STORAGE, SlotRole.STORAGE),
        2,
        2,
        MAX_HEALTH,
    ),
)


def _validate(specs: tuple[MachineSpec, ...]) -> None:
    """Validate the spec table at import time.

    Parameters
    ----------
    specs :
        The full machine spec table.

    ...] :

    Raises
    ------
    ValueError
        If the table does not hold exactly one record per
        ``Machine`` in value order, or a record has a negative
        capacity.
    """
    if len(specs) != len(Machine):
        raise ValueError(
            f"MACHINE_SPECS has {len(specs)} records; expected one per "
            f"Machine ({len(Machine)})."
        )
    for i, spec in enumerate(specs):
        if spec.machine_type != Machine(i):
            raise ValueError(
                f"MACHINE_SPECS[{i}] defines {spec.machine_type!r}; expected "
                f"{Machine(i)!r} (records must be in Machine order)."
            )
        if min(spec.buffer_stack, spec.max_types, spec.max_health) < 0:
            raise ValueError(f"{spec.machine_type!r} has a negative capacity.")


_validate(MACHINE_SPECS)

# --- Derived arrays — the engine and editor index these --------------------

#: Logical slot count per machine. numpy int32 to match the editor.
MACHINE_NUM_SLOTS = np.array([s.num_slots for s in MACHINE_SPECS], dtype=np.int32)

#: Editor slot-view width: the largest slot count across all machines.
MAX_MACHINE_INVENTORY_SLOTS: int = int(MACHINE_NUM_SLOTS.max())

#: Per-slot role per machine, padded to the view width with NONE.
MACHINE_SLOT_ROLES = np.zeros(
    (len(MACHINE_SPECS), MAX_MACHINE_INVENTORY_SLOTS), dtype=np.int32
)
for _i, _spec in enumerate(MACHINE_SPECS):
    for _j, _role in enumerate(_spec.slots):
        MACHINE_SLOT_ROLES[_i, _j] = int(_role)

#: Per-machine buffer capacity. jnp int16 (the engine reads it as int16).
MACHINE_MAX_STACK = jnp.array([s.buffer_stack for s in MACHINE_SPECS], dtype=jnp.int16)

#: Distinct item types each buffer may hold. jnp int32.
MACHINE_MAX_TYPES = jnp.array([s.max_types for s in MACHINE_SPECS], dtype=jnp.int32)

#: Maximum health per machine type. jnp int16.
MACHINE_MAX_HEALTH = jnp.array([s.max_health for s in MACHINE_SPECS], dtype=jnp.int16)
