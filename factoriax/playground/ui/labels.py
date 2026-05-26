"""Human-readable display names for FactoriaX UI applications.

Maps engine enum values to the label strings the editor and play UI
render in panel headers, tooltips, and toolbars. Presentation only —
kept out of the core :mod:`factoriax.engine.constants` module.
"""

from __future__ import annotations

from factoriax.engine.constants import Machine

# Human-readable display names for each Machine, derived from the
# enum: underscores become spaces and each word is title-cased, giving
# e.g. "Conveyor Belt". Every member is covered, so a new Machine
# gets a name automatically. Mirrors the _ITEM_NAMES derivation in
# editor/inventory_panel.py. A machine needing a name that differs from
# its title-cased enum identifier would require an override layered on
# top of these defaults.
MACHINE_TYPE_NAMES: dict[int, str] = {
    int(mt): mt.name.replace("_", " ").title() for mt in Machine
}
