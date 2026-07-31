"""Human-readable display names for FactoriaX UI applications.

Maps engine enum values to the label strings the editor and play UI
render in panel headers, tooltips, and toolbars. Presentation only —
kept out of the core :mod:`factoriax.engine.constants` module.
"""

from __future__ import annotations

from factoriax.engine.constants import Machine

# Human-readable display names for each Machine, derived from the
# enum: an underscore becomes a space, and each word is title-cased, for
# example "Conveyor Belt". Every member gets a name, so a new Machine needs no
# edit here. The _ITEM_NAMES tables build their names the same way. A machine
# that needs another name takes an override on top of this table.
MACHINE_TYPE_NAMES: dict[int, str] = {
    int(mt): mt.name.replace("_", " ").title() for mt in Machine
}
