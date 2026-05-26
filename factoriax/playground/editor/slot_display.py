"""Display data for machine-inventory slot-role badges in the editor.

Maps :class:`~factoriax.engine.constants.SlotRole` integer values to the badge
label text and color the editor draws for each machine inventory slot.
This is presentation only — the role *semantics* live with the
``SlotRole`` enum and ``MACHINE_SLOT_ROLES`` in
:mod:`factoriax.engine.constants`.
"""

from __future__ import annotations

SLOT_ROLE_LABELS: dict[int, str] = {
    0: "",
    1: "IN",
    2: "OUT",
    3: "STORE",
    4: "FUEL",
}

SLOT_ROLE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),
    1: (190, 120, 40),
    2: (40, 170, 140),
    3: (80, 115, 175),
    4: (200, 60, 60),
}
