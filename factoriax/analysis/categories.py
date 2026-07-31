"""Group the engine's actions and items into colored categories.

Plot code asks two questions of this module. Which category does this
action or item belong to, and what color is that category? The answers
drive the bands of a stacked area chart and the node fills of the
recipe diagram.

Every member of a category shares the base color of that category. The
color therefore gives the category, and the label gives the member.
There is no separate shade for each member.

The action groups are written out by hand, because a category such as
"Movement" has no enum of its own. The item groups come from the
engine's ``Resource``, ``HalfFabricate``, and ``Machine`` enums, so
they cannot drift from the item table in ``constants.py``.

Every key in the returned maps is the upper-case name of an
:class:`~factoriax.engine.constants.ItemType` or
:class:`~factoriax.engine.constants.Action` member, such as
``IRON_PLATE``. Colors are hex strings such as ``"#2ca02c"``.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import IntEnum

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

# Each item category is one of the engine's source enums. The label
# and base color are set here. The membership comes from the enum, so
# this file never drifts from the item table in constants.py.
_CATEGORY_SOURCES: tuple[tuple[str, str, type[IntEnum]], ...] = (
    ("Machines", "#2ca02c", Machine),
    ("Half Fabricates", "#ff7f0e", HalfFabricate),
    ("Resources", "#8c564b", Resource),
)


def _build_item_groups() -> OrderedDict[str, tuple[str, list[int]]]:
    """Build the item groups from the engine's item enums.

    Returns
    -------
    collections.OrderedDict
        ``{category_label: (base_hex, [item_type_values])}``, in the
        order of :data:`_CATEGORY_SOURCES`. The ``NONE`` member of each
        enum is dropped, because it names no item.
    """
    groups: OrderedDict[str, tuple[str, list[int]]] = OrderedDict()
    for label, base_hex, enum in _CATEGORY_SOURCES:
        members = [
            int(ItemType[member.name]) for member in enum if member.name != "NONE"
        ]
        groups[label] = (base_hex, members)
    return groups


ITEM_GROUPS: OrderedDict[str, tuple[str, list[int]]] = _build_item_groups()


def item_palette() -> dict[str, str]:
    """Map every item name to the fill color of its category.

    Every item in a category shares the base color of that category.
    The color therefore gives the category and nothing more. The label
    separates one item from another. There is no shade for each item,
    and this is deliberate. A reader cannot rank twelve shades of one
    hue, but a reader can read a name.

    Returns
    -------
    dict
        ``{item_name: hex}``. The key is the upper-case ``ItemType``
        member name, such as ``IRON_PLATE``. An item outside the three
        category enums is absent, so the caller needs a fallback
        color.
    """
    palette: dict[str, str] = {}
    for _name, (base_hex, members) in ITEM_GROUPS.items():
        for item in members:
            palette[ItemType(item).name] = base_hex
    return palette


def category_palette() -> dict[str, str]:
    """Map every item category name to its base color.

    Use this for a legend, where one entry stands for a whole category.
    Use :func:`item_palette` to color a single item.

    Returns
    -------
    dict
        ``{category_label: hex}``, such as ``{"Machines": "#2ca02c"}``.
        The insertion order follows :data:`_CATEGORY_SOURCES`, so a
        legend built from it keeps a stable order between runs.
    """
    return {name: base_hex for name, (base_hex, _) in ITEM_GROUPS.items()}


def item_to_category() -> dict[str, str]:
    """Map every item name to the name of its category.

    This is the inverse view of :data:`ITEM_GROUPS`. Use it to decide
    which legend entry an item belongs under.

    Returns
    -------
    dict
        ``{item_name: category_label}``. The key is the upper-case
        ``ItemType`` member name, such as ``IRON_PLATE``. An item
        outside the three category enums is absent, so a caller must
        handle a missing key.
    """
    mapping: dict[str, str] = {}
    for name, (_, members) in ITEM_GROUPS.items():
        for item in members:
            mapping[ItemType(item).name] = name
    return mapping
