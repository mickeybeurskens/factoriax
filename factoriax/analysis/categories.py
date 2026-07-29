"""Action and item category definitions with shade palettes.

Groups actions and items into semantic categories, each with a base
color. Individual members within a group get distinct shades of that
base color, producing the banded look of an Age-of-Empires-style
stacked area chart.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import IntEnum

import matplotlib.colors as mcolors
import numpy as np

from factoriax.engine.constants import (
    Action,
    HalfFabricate,
    ItemType,
    Machine,
    Resource,
)

# ---------------------------------------------------------------
# Action groups: (base_hex, [Action members])
# ---------------------------------------------------------------

ACTION_GROUPS: OrderedDict[str, tuple[str, list[int]]] = OrderedDict(
    [
        (
            "Movement",
            (
                "#2ca02c",
                [
                    Action.NOOP,
                    Action.UP,
                    Action.DOWN,
                    Action.LEFT,
                    Action.RIGHT,
                    Action.FACE_UP,
                    Action.FACE_DOWN,
                    Action.FACE_LEFT,
                    Action.FACE_RIGHT,
                ],
            ),
        ),
        (
            "Mining",
            ("#d62728", [Action.MINE, Action.PICKUP]),
        ),
        (
            "Placement",
            (
                "#1f77b4",
                [
                    Action.PLACE_MINER,
                    Action.PLACE_PALLET,
                    Action.PLACE_CONVEYOR_BELT,
                    Action.PLACE_ASSEMBLER,
                    Action.PLACE_ROCKET,
                ],
            ),
        ),
        (
            "Crafting",
            (
                "#9467bd",
                [
                    Action.CRAFT_IRON_PLATE,
                    Action.CRAFT_COPPER_PLATE,
                    Action.CRAFT_TIN_PLATE,
                    Action.CRAFT_WAFER,
                    Action.CRAFT_FRAME,
                    Action.CRAFT_CIRCUIT,
                    Action.CRAFT_WIRE,
                    Action.CRAFT_MOTOR,
                    Action.CRAFT_SENSOR,
                    Action.CRAFT_CONVEYOR_BELT,
                    Action.CRAFT_MINER,
                    Action.CRAFT_ASSEMBLER,
                    Action.CRAFT_PALLET,
                    Action.CRAFT_TIER1_SCIENCE_PACK,
                    Action.CRAFT_TIER2_SCIENCE_PACK,
                    Action.CRAFT_ROCKET,
                ],
            ),
        ),
        (
            "Deposit",
            (
                "#17becf",
                [
                    Action.DEPOSIT_COAL,
                    Action.DEPOSIT_IRON_ORE,
                    Action.DEPOSIT_COPPER_ORE,
                    Action.DEPOSIT_TIN_ORE,
                    Action.DEPOSIT_SILICON,
                    Action.DEPOSIT_IRON_PLATE,
                    Action.DEPOSIT_COPPER_PLATE,
                    Action.DEPOSIT_TIN_PLATE,
                    Action.DEPOSIT_WAFER,
                    Action.DEPOSIT_FRAME,
                    Action.DEPOSIT_CIRCUIT,
                    Action.DEPOSIT_WIRE,
                    Action.DEPOSIT_MOTOR,
                    Action.DEPOSIT_SENSOR,
                    Action.DEPOSIT_CONVEYOR_BELT,
                    Action.DEPOSIT_MINER,
                    Action.DEPOSIT_ASSEMBLER,
                    Action.DEPOSIT_PALLET,
                    Action.DEPOSIT_TIER1_SCIENCE_PACK,
                    Action.DEPOSIT_TIER2_SCIENCE_PACK,
                    Action.DEPOSIT_ROCKET,
                    Action.DEPOSIT_FURNACE,
                    Action.DEPOSIT_REFRACTORY,
                    Action.DEPOSIT_HULL,
                    Action.DEPOSIT_ENGINE_UNIT,
                    Action.DEPOSIT_AVIONICS,
                    Action.DEPOSIT_ROCKET_CORE,
                ],
            ),
        ),
        (
            "Withdraw",
            (
                "#8c564b",
                [Action.WITHDRAW],
            ),
        ),
        (
            "Rotate",
            (
                "#7f7f7f",
                [
                    Action.ROTATE_LEFT,
                    Action.ROTATE_RIGHT,
                    Action.ROTATE_UP,
                    Action.ROTATE_DOWN,
                    Action.REPAIR,
                ],
            ),
        ),
    ]
)

# ---------------------------------------------------------------
# Item groups: (base_hex, [ItemType members])
# ---------------------------------------------------------------

# Each item category is one of the engine's source enums. The visual
# label and base colour are paper-side concerns; the membership comes
# straight from the engine so categories.py never drifts from the item
# table in constants.py.
_CATEGORY_SOURCES: tuple[tuple[str, str, type[IntEnum]], ...] = (
    ("Machines", "#2ca02c", Machine),
    ("Half Fabricates", "#ff7f0e", HalfFabricate),
    ("Resources", "#8c564b", Resource),
)


def _build_item_groups() -> OrderedDict[str, tuple[str, list[int]]]:
    """Mirror the engine's ``Resource``/``HalfFabricate``/``Machine`` enums."""
    groups: OrderedDict[str, tuple[str, list[int]]] = OrderedDict()
    for label, base_hex, enum in _CATEGORY_SOURCES:
        members = [
            int(ItemType[member.name]) for member in enum if member.name != "NONE"
        ]
        groups[label] = (base_hex, members)
    return groups


ITEM_GROUPS: OrderedDict[str, tuple[str, list[int]]] = _build_item_groups()


def shade_palette(
    base_hex: str,
    n: int,
) -> list[str]:
    """Generate ``n`` shades of a base color from dark to light.

    Interpolates lightness between 60% and 130% of the base RGB,
    clamped to [0, 1]. With one member the base color is returned
    unchanged.

    Parameters
    ----------
    base_hex :
        Base color as a hex string (e.g. ``"#2ca02c"``).
    n :
        Number of shades to produce.
    base_hex : str :

    n : int :

    base_hex: str :

    n: int :


    Returns
    -------


    """
    if n <= 0:
        return []
    if n == 1:
        return [base_hex]
    rgb = np.array(mcolors.to_rgb(base_hex))
    factors = np.linspace(0.6, 1.3, n)
    return [mcolors.to_hex(np.clip(rgb * f, 0.0, 1.0)) for f in factors]


def item_palette() -> dict[str, str]:
    """ """
    palette: dict[str, str] = {}
    for _name, (base_hex, members) in ITEM_GROUPS.items():
        for item in members:
            palette[ItemType(item).name] = base_hex
    return palette


def category_palette() -> dict[str, str]:
    """ """
    return {name: base_hex for name, (base_hex, _) in ITEM_GROUPS.items()}


def item_to_category() -> dict[str, str]:
    """ """
    mapping: dict[str, str] = {}
    for name, (_, members) in ITEM_GROUPS.items():
        for item in members:
            mapping[ItemType(item).name] = name
    return mapping
