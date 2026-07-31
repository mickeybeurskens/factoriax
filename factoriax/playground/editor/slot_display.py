"""The slot layout that the editor draws for each machine.

The module gives the number of slots for each machine kind. It also gives the
part that each slot has in the recipe, with the label and the color of the
badge for that part.

None of this reaches the simulation. A level records the contents of a machine
by item, and records no slot. A slot is therefore a view that the editor
draws, and :mod:`factoriax.playground.editor.state` builds that view from the
items.

:mod:`factoriax.engine.levels` gives each item a slot by its own rules, and
never reads the layout here. The meaning of each part is with
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
