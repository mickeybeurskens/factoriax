"""Machine inventory slot layout and role badges for the editor.

Declares how many inventory slots the editor shows per machine kind, what role
each slot has, and the badge label and colour drawn for a role.

None of this reaches the simulation. A level file stores machine contents as an
item-indexed pouch, so slots are an editor view that
:mod:`factoriax.playground.editor.state` converts to on load and back from on
save. :mod:`factoriax.engine.levels` packs the pouch into engine arrays with its
own per-machine rules and never reads the layout here. Role semantics live with
:class:`~factoriax.engine.constants.SlotRole`.
"""

from __future__ import annotations

import numpy as np

from factoriax.engine.constants import Machine, SlotRole

# Slot roles per machine kind, in the order the editor draws them. Iterating
# Machine to build the arrays below means a kind added to the enum without an
# entry here raises KeyError at import instead of drawing no slots.
MACHINE_SLOTS: dict[Machine, tuple[SlotRole, ...]] = {
    Machine.NONE: (),
    Machine.MINER: (SlotRole.OUTPUT,),
    Machine.PALLET: (SlotRole.STORAGE,),
    Machine.CONVEYOR_BELT: (SlotRole.STORAGE,),
    Machine.ASSEMBLER: (SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT),
    Machine.ARM: (),
    Machine.ROCKET: (),
    Machine.FURNACE: (SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT),
    Machine.SCIENCE_LAB: (SlotRole.INPUT, SlotRole.INPUT),
    Machine.SPLITTER: (SlotRole.STORAGE,),
    Machine.CROSSING: (SlotRole.STORAGE, SlotRole.STORAGE),
}

#: Slot count per machine, indexed by ``Machine`` value. Shape
#: ``(len(Machine),)``, numpy int32. Zero for a machine the editor shows no
#: slots for, including ``ARM``, which carries an item in transit rather than
#: storing one.
MACHINE_NUM_SLOTS = np.array(
    [len(MACHINE_SLOTS[m]) for m in Machine],
    dtype=np.int32,
)

#: Slot count of the widest machine. Fixes the row width of
#: :data:`MACHINE_SLOT_ROLES` and of the editor's per-tile slot arrays, which
#: are rectangular and so pad every machine out to this many slots.
MAX_MACHINE_INVENTORY_SLOTS: int = int(MACHINE_NUM_SLOTS.max())

#: Slot roles per machine, shape ``(len(Machine), MAX_MACHINE_INVENTORY_SLOTS)``,
#: numpy int32 holding :class:`~factoriax.engine.constants.SlotRole` values.
#: Columns past a machine's slot count are ``SlotRole.NONE``, marking a slot
#: that does not exist.
MACHINE_SLOT_ROLES = np.zeros(
    (len(Machine), MAX_MACHINE_INVENTORY_SLOTS),
    dtype=np.int32,
)
for _machine, _roles in MACHINE_SLOTS.items():
    for _slot, _role in enumerate(_roles):
        MACHINE_SLOT_ROLES[int(_machine), _slot] = int(_role)

#: Badge text per :class:`~factoriax.engine.constants.SlotRole` value. ``NONE``
#: maps to the empty string, so a padding slot draws no badge.
SLOT_ROLE_LABELS: dict[int, str] = {
    0: "",
    1: "IN",
    2: "OUT",
    3: "STORE",
    4: "FUEL",
}

#: Badge fill colour per :class:`~factoriax.engine.constants.SlotRole` value, as
#: RGB in 0..255.
SLOT_ROLE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),
    1: (190, 120, 40),
    2: (40, 170, 140),
    3: (80, 115, 175),
    4: (200, 60, 60),
}
