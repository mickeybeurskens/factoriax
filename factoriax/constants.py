"""Constants and enumerations for the FactoriaX environment.

Pure-Python definitions only. Their derived JAX arrays (gather tables, the
state-array dtypes) live in :mod:`factoriax.tables`.
"""

from enum import IntEnum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BlockType(IntEnum):
    """Block types in the environment grid."""

    INVALID = 0
    OUT_OF_BOUNDS = 1
    DIRT = 2
    WATER = 3
    IRON = 4
    COPPER = 5
    COAL = 6
    TIN = 7
    SILICON = 8
    LIMESTONE = 9


class Direction(IntEnum):
    """Compass facing directions for players and machines."""

    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class SlotRole(IntEnum):
    """Slot roles for machine inventory display (editor compat)."""

    NONE = 0
    INPUT = 1
    OUTPUT = 2
    STORAGE = 3
    FUEL = 4


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

# The three item categories. Each carries a NONE=0 placeholder, stripped from
# every derivative, so the categories share a uniform shape and Machine can
# double as the entity tag. ItemType is composed from their member names.


class Resource(IntEnum):
    """Mineable raw materials."""

    NONE = 0
    COAL = 1
    IRON_ORE = 2
    COPPER_ORE = 3
    TIN_ORE = 4
    SILICON = 5
    LIMESTONE = 6


class HalfFabricate(IntEnum):
    """Crafted, non-placeable intermediates."""

    NONE = 0
    IRON_PLATE = 1
    COPPER_PLATE = 2
    TIN_PLATE = 3
    WAFER = 4
    FRAME = 5
    CIRCUIT = 6
    WIRE = 7
    MOTOR = 8
    SENSOR = 9
    BASIC_SCIENCE_PACK = 10
    ADVANCED_SCIENCE_PACK = 11
    REFRACTORY = 12
    HULL = 13
    ENGINE_UNIT = 14
    AVIONICS = 15
    ROCKET_CORE = 16


class Machine(IntEnum):
    """Placeable machine kinds and the entity tag in ``ent_type`` /
    ``machine_types``.

    Its names compose ItemType and the action families; its values are the
    tile/entity tag. ``NONE = 0`` is the empty cell, the analog of
    ``ItemType.EMPTY = 0``.
    """

    NONE = 0
    MINER = 1
    PALLET = 2
    CONVEYOR_BELT = 3
    ASSEMBLER = 4
    ARM = 5
    ROCKET = 6
    FURNACE = 7
    SCIENCE_LAB = 8
    SPLITTER = 9
    CROSSING = 10


_ITEM_CATEGORIES: tuple[type[IntEnum], ...] = (Resource, HalfFabricate, Machine)
ItemType = IntEnum(
    "ItemType",
    [
        "EMPTY",
        *(m.name for cat in _ITEM_CATEGORIES for m in cat if m.name != "NONE"),
    ],
    start=0,
)
ItemType.__doc__ = "Item types that can be stored in inventory."


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class MoveAction(IntEnum):
    """Movement and facing. A fixed action family."""

    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    FACE_UP = 5
    FACE_DOWN = 6
    FACE_LEFT = 7
    FACE_RIGHT = 8


class InteractAction(IntEnum):
    """Non-parametric world interactions. A fixed action family."""

    MINE = 0
    PICKUP = 1
    WITHDRAW = 2
    REPAIR = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    ROTATE_UP = 6
    ROTATE_DOWN = 7


# Items each parametric action family addresses (NONE stripped): Placement per
# Machine, Craft per non-resource item, Deposit per non-EMPTY item. Action and
# the game_logic offset->item tables both derive from these.
PLACEMENT_ITEMS: tuple[Machine, ...] = tuple(m for m in Machine if m.name != "NONE")
CRAFT_ITEMS: tuple[HalfFabricate | Machine, ...] = tuple(
    m for cat in (HalfFabricate, Machine) for m in cat if m.name != "NONE"
)
DEPOSIT_ITEMS: tuple[Resource | HalfFabricate | Machine, ...] = tuple(
    m for cat in (Resource, HalfFabricate, Machine) for m in cat if m.name != "NONE"
)

_ACTION_NAMES: list[str] = [
    *(a.name for a in MoveAction),
    *(a.name for a in InteractAction),
    *(f"PLACE_{m.name}" for m in PLACEMENT_ITEMS),
    *(f"CRAFT_{m.name}" for m in CRAFT_ITEMS),
    *(f"DEPOSIT_{m.name}" for m in DEPOSIT_ITEMS),
]
Action = IntEnum("Action", _ACTION_NAMES, start=0)
Action.__doc__ = """Player actions using compound action design.

Each action is self-contained: placement, deposit, and withdraw actions name
the item type, so no slot cursor is needed. Movement actions move in absolute
map directions; FACE_* snaps facing without moving.
"""


# ---------------------------------------------------------------------------
# Derived sizes and offsets
# ---------------------------------------------------------------------------

NUM_ITEM_TYPES = len(ItemType)
NUM_ACTIONS = len(Action)
MAX_ACHIEVEMENTS = 64
BLOCK_MAX_RESOURCES = 30000

# Base offsets for the parametric action families. The PLACE_/CRAFT_/DEPOSIT_
# offset->item tables live with the dispatcher in factoriax.actions.
PLACE_BASE: int = len(MoveAction) + len(InteractAction)
CRAFT_BASE: int = PLACE_BASE + len(PLACEMENT_ITEMS)
DEPOSIT_BASE: int = CRAFT_BASE + len(CRAFT_ITEMS)
ROTATE_BASE: int = len(MoveAction) + int(InteractAction.ROTATE_LEFT)


# ---------------------------------------------------------------------------
# Item / machine / block mappings
# ---------------------------------------------------------------------------

# The item<->machine bijection and anchor for the machine cluster: every
# placeable item and the machine it becomes. The placeable list and the
# factoriax.placement gather arrays are projections of it.
ITEM_TO_MACHINE = {
    ItemType.MINER: Machine.MINER,
    ItemType.PALLET: Machine.PALLET,
    ItemType.CONVEYOR_BELT: Machine.CONVEYOR_BELT,
    ItemType.ASSEMBLER: Machine.ASSEMBLER,
    ItemType.ARM: Machine.ARM,
    ItemType.ROCKET: Machine.ROCKET,
    ItemType.FURNACE: Machine.FURNACE,
    ItemType.SCIENCE_LAB: Machine.SCIENCE_LAB,
    ItemType.SPLITTER: Machine.SPLITTER,
    ItemType.CROSSING: Machine.CROSSING,
}

# Placeable items: the mapping's keys. Order is incidental (it only sets the
# play-UI palette order; the agent resolves placement by item identity).
PLACEABLE_ITEM_LIST: tuple[int, ...] = tuple(int(it) for it in ITEM_TO_MACHINE)

RESOURCE_ITEM_LIST: tuple[int, ...] = tuple(
    i for i in range(1, NUM_ITEM_TYPES) if i not in PLACEABLE_ITEM_LIST
)

# Ore block -> mined item.
BLOCK_TO_ITEM: dict[int, int] = {
    BlockType.COAL: ItemType.COAL,
    BlockType.IRON: ItemType.IRON_ORE,
    BlockType.COPPER: ItemType.COPPER_ORE,
    BlockType.TIN: ItemType.TIN_ORE,
    BlockType.SILICON: ItemType.SILICON,
    BlockType.LIMESTONE: ItemType.LIMESTONE,
}

# Science packs consumed by SCIENCE_LAB entities, indexed by position into the
# per-step delta vector ``EnvState.science_consumed_step``.
SCIENCE_PACK_TYPES: tuple[int, ...] = (
    int(ItemType.BASIC_SCIENCE_PACK),
    int(ItemType.ADVANCED_SCIENCE_PACK),
)
NUM_SCIENCE_PACK_TYPES: int = len(SCIENCE_PACK_TYPES)
