"""Every placeable machine must have a renderable icon.

Catches the case where a new MachineType is added to ITEM_TO_MACHINE
in constants.py but the renderer's MACHINE_TO_ITEM or render_item_icon
doesn't know about it. The arm was invisible on the map because of
exactly this kind of disconnect.
"""

from __future__ import annotations

from factoriax.constants import ITEM_TO_MACHINE, Direction, MachineType
from factoriax.renderer import MACHINE_TO_ITEM, render_item_icon


class TestMachineIconCoverage:
    """Every machine in ITEM_TO_MACHINE must render without error."""

    def test_all_machines_in_renderer_dict(self) -> None:
        """MACHINE_TO_ITEM covers every MachineType from constants."""
        for item_type, machine_type in ITEM_TO_MACHINE.items():
            assert int(machine_type) in MACHINE_TO_ITEM, (
                f"MachineType {machine_type!r} (from ItemType "
                f"{item_type!r}) missing from renderer.MACHINE_TO_ITEM"
            )

    def test_all_machines_render_icon(self) -> None:
        """render_item_icon produces a valid array for every machine."""
        for item_type in ITEM_TO_MACHINE:
            icon = render_item_icon(
                int(item_type),
                32,
                int(Direction.RIGHT),
            )
            assert icon.shape == (32, 32, 4), (
                f"ItemType {item_type!r}: bad icon shape {icon.shape}"
            )
            assert icon.dtype.name == "uint8"

    def test_machine_type_none_excluded(self) -> None:
        """MachineType.NONE should not appear in the mapping."""
        assert int(MachineType.NONE) not in MACHINE_TO_ITEM
