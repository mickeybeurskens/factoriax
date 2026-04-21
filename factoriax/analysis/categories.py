"""Action and item category definitions with shade palettes.

Groups actions and items into semantic categories, each with a base
color. Individual members within a group get distinct shades of that
base color, producing the banded look of an Age-of-Empires-style
stacked area chart.
"""

from __future__ import annotations

from collections import OrderedDict

import matplotlib.colors as mcolors
import numpy as np

from factoriax.constants import Action, ItemType

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
                    Action.PLACE_BELT,
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
                    Action.CRAFT_BELT,
                    Action.CRAFT_MINER,
                    Action.CRAFT_ASSEMBLER,
                    Action.CRAFT_PALLET,
                    Action.CRAFT_BASIC_SCIENCE,
                    Action.CRAFT_ADV_SCIENCE,
                    Action.CRAFT_ROCKET,
                ],
            ),
        ),
        (
            "Research",
            (
                "#bcbd22",
                [Action.RESEARCH_BASIC, Action.RESEARCH_ADVANCED],
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
                    Action.DEPOSIT_BELT,
                    Action.DEPOSIT_MINER,
                    Action.DEPOSIT_ASSEMBLER,
                    Action.DEPOSIT_PALLET,
                    Action.DEPOSIT_BASIC_SCIENCE,
                    Action.DEPOSIT_ADV_SCIENCE,
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
            "Legacy",
            (
                "#7f7f7f",
                [
                    Action.TURN_LEFT,
                    Action.TURN_RIGHT,
                    Action.ROTATE,
                    Action.REPAIR,
                ],
            ),
        ),
    ]
)

# ---------------------------------------------------------------
# Item groups: (base_hex, [ItemType members])
# ---------------------------------------------------------------

ITEM_GROUPS: OrderedDict[str, tuple[str, list[int]]] = OrderedDict(
    [
        (
            "Ores",
            (
                "#8c564b",
                [
                    ItemType.COAL,
                    ItemType.IRON_ORE,
                    ItemType.COPPER_ORE,
                    ItemType.TIN_ORE,
                    ItemType.SILICON,
                ],
            ),
        ),
        (
            "Plates",
            (
                "#aaa9ad",
                [
                    ItemType.IRON_PLATE,
                    ItemType.COPPER_PLATE,
                    ItemType.TIN_PLATE,
                ],
            ),
        ),
        (
            "Intermediates",
            (
                "#1f77b4",
                [
                    ItemType.WAFER,
                    ItemType.FRAME,
                    ItemType.CIRCUIT,
                    ItemType.WIRE,
                    ItemType.MOTOR,
                    ItemType.SENSOR,
                ],
            ),
        ),
        (
            "Science",
            (
                "#d62728",
                [
                    ItemType.BASIC_SCIENCE_PACK,
                    ItemType.ADVANCED_SCIENCE_PACK,
                ],
            ),
        ),
        (
            "Machines",
            (
                "#2ca02c",
                [
                    ItemType.CONVEYOR_BELT,
                    ItemType.MINER,
                    ItemType.ASSEMBLER,
                    ItemType.PALLET,
                    ItemType.ROCKET,
                ],
            ),
        ),
    ]
)


def shade_palette(
    base_hex: str,
    n: int,
) -> list[str]:
    """Generate ``n`` shades of a base color from dark to light.

    Interpolates lightness between 60% and 130% of the base RGB,
    clamped to [0, 1]. With one member the base color is returned
    unchanged.

    Args:
        base_hex: Base color as a hex string (e.g. ``"#2ca02c"``).
        n: Number of shades to produce.

    Returns:
        List of hex color strings.
    """
    if n <= 0:
        return []
    if n == 1:
        return [base_hex]
    rgb = np.array(mcolors.to_rgb(base_hex))
    factors = np.linspace(0.6, 1.3, n)
    return [mcolors.to_hex(np.clip(rgb * f, 0.0, 1.0)) for f in factors]
