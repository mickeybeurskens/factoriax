"""Test that the editor toolbar handles underground belt machine types."""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

pygame.init()
pygame.display.set_mode((100, 100))

from factoriax.constants import MachineType
from factoriax.editor.toolbar import MACHINE_TO_ITEM_MAP


class TestMachineToItemMap:
    """MACHINE_TO_ITEM_MAP must cover all non-NONE machine types."""

    def test_all_machine_types_have_item_mapping(self) -> None:
        """Every MachineType except NONE must be in MACHINE_TO_ITEM_MAP."""
        for mt in MachineType:
            if mt == MachineType.NONE:
                continue
            assert int(mt) in MACHINE_TO_ITEM_MAP, (
                f"MachineType.{mt.name} ({int(mt)}) missing from "
                f"MACHINE_TO_ITEM_MAP"
            )
