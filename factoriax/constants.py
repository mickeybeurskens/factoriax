"""Constants and enumerations for the FactoriaX environment."""

from enum import IntEnum

import jax.numpy as jnp


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
    LIMESTONE = 9  # Refractory feedstock; pairs with COAL in furnace


# ItemType is composed from the item categories below (EMPTY + Resource +
# HalfFabricate + Machine); see its definition just after the Machine class.


# ---------------------------------------------------------------------------
# Item categories (DEF2)
# ---------------------------------------------------------------------------
# Every non-EMPTY item belongs to exactly one category. These three enums are
# the definitions; DEF2 composes ItemType from them (EMPTY + Resource +
# HalfFabricate + Machine). Their own integer values are inert -- the real
# item id lives on ItemType -- so only the member names and the grouping
# matter. A test (test_item_categories) guards that they partition ItemType.


class Resource(IntEnum):
    """Mineable raw materials. Values inert (the item id is on ItemType).

    ``NONE = 0`` is a placeholder so every item category shares the
    "NONE=0, real members from 1" shape; it is stripped from all derivatives.
    """

    NONE = 0
    COAL = 1
    IRON_ORE = 2
    COPPER_ORE = 3
    TIN_ORE = 4
    SILICON = 5
    LIMESTONE = 6


class HalfFabricate(IntEnum):
    """Crafted, non-placeable intermediates. Values inert (see Resource)."""

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
    """Placeable machine kinds -- the single machine enum.

    Two roles in one: its *names* compose ItemType and the PLACE/CRAFT/DEPOSIT
    action families (NONE stripped, values never leak there), and its *values*
    are the entity tag stored in ``ent_type`` / ``machine_types``. ``NONE = 0``
    is the empty-cell tag -- the machine-space analog of ``ItemType.EMPTY = 0``
    -- so zero-filled tag arrays read as empty and ``MACHINE_SPECS[0]`` is the
    empty row.
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
    # Belt-network pieces. SPLITTER reuses ent_buf (stack=2) and outputs to the
    # two perpendicular sides of its facing direction. CROSSING reuses
    # ent_asm_in[0..1] for two independent per-axis buffers so the vertical and
    # horizontal streams co-exist on one tile without mixing.
    SPLITTER = 9
    CROSSING = 10


# ---------------------------------------------------------------------------
# ItemType -- composed from the item categories (DEF2)
# ---------------------------------------------------------------------------
# EMPTY, then the three categories in order: contiguous and renumbered. The
# category enums are the single source -- nothing here hand-assigns a flat item
# id. Existing ``ItemType.X`` references, ``int(ItemType.X)``, ``ItemType(i)``,
# ``.name`` and iteration all resolve at runtime; only the integer *values*
# move. That is a deliberate break: the observation encodes item ids as
# normalized scalars, so renumbering changes what a trained agent perceives and
# invalidates saved trajectories.
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


NUM_ITEM_TYPES = len(ItemType)

# Canonical dtypes for state arrays.
INVENTORY_COUNT_DTYPE = jnp.int32
MACHINE_INVENTORY_COUNT_DTYPE = jnp.int16
BLOCK_RESOURCE_DTYPE = jnp.int16

# Max stack count per item type for the player inventory.
# Per-item player inventory stack cap, indexed by ItemType. Most items
# stack to a common default; machines and large rocket components are
# bulky, so the player carries fewer per stack. Built by name-keyed
# overrides over the default so a row can never drift out of alignment
# with ItemType the way a positional literal can.
_DEFAULT_PLAYER_STACK = 1024
_BULKY_PLAYER_STACK = 128
_PLAYER_STACK_OVERRIDES: dict[int, int] = {
    int(ItemType.EMPTY): 0,
    int(ItemType.CONVEYOR_BELT): _BULKY_PLAYER_STACK,
    int(ItemType.MINER): _BULKY_PLAYER_STACK,
    int(ItemType.ASSEMBLER): _BULKY_PLAYER_STACK,
    int(ItemType.PALLET): _BULKY_PLAYER_STACK,
    int(ItemType.ARM): _BULKY_PLAYER_STACK,
    int(ItemType.ROCKET): _BULKY_PLAYER_STACK,
    int(ItemType.FURNACE): _BULKY_PLAYER_STACK,
    int(ItemType.HULL): _BULKY_PLAYER_STACK,
    int(ItemType.ENGINE_UNIT): _BULKY_PLAYER_STACK,
    int(ItemType.AVIONICS): _BULKY_PLAYER_STACK,
    int(ItemType.ROCKET_CORE): _BULKY_PLAYER_STACK,
    int(ItemType.SCIENCE_LAB): _BULKY_PLAYER_STACK,
    int(ItemType.SPLITTER): _BULKY_PLAYER_STACK,
    int(ItemType.CROSSING): _BULKY_PLAYER_STACK,
}
PLAYER_MAX_STACK = jnp.array(
    [
        _PLAYER_STACK_OVERRIDES.get(i, _DEFAULT_PLAYER_STACK)
        for i in range(NUM_ITEM_TYPES)
    ],
    dtype=jnp.int32,
)

BLOCK_TO_ITEM: dict[int, int] = {
    BlockType.COAL: ItemType.COAL,
    BlockType.IRON: ItemType.IRON_ORE,
    BlockType.COPPER: ItemType.COPPER_ORE,
    BlockType.TIN: ItemType.TIN_ORE,
    BlockType.SILICON: ItemType.SILICON,
    BlockType.LIMESTONE: ItemType.LIMESTONE,
}

# ---------------------------------------------------------------------------
# Science packs
# ---------------------------------------------------------------------------

# Tracked per-type by SCIENCE_LAB entities. The per-step delta vector
# in EnvState.science_consumed_step is sized to NUM_SCIENCE_PACK_TYPES
# and indexed by position in SCIENCE_PACK_TYPES. The jnp item->index
# projection is built in factoriax.game_logic, its sole engine consumer.
SCIENCE_PACK_TYPES: tuple[int, ...] = (
    int(ItemType.BASIC_SCIENCE_PACK),
    int(ItemType.ADVANCED_SCIENCE_PACK),
)
NUM_SCIENCE_PACK_TYPES: int = len(SCIENCE_PACK_TYPES)

# ---------------------------------------------------------------------------
# Placeable items and machine mappings
# ---------------------------------------------------------------------------

# The item<->machine bijection: every placeable item and the machine it
# becomes when placed. This is the anchor for the machine cluster -- the
# placeable set below and the jnp gather arrays in factoriax.placement are
# all projections of it.
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

# Items that place a machine when used -- exactly the keys of the mapping
# above, so the two cannot drift. This is a membership definition; its
# sequence order is incidental (it inherits the mapping's key order). The
# agent action layer (PLACE_ACTION_TO_ITEM in factoriax.actions) is independent
# and resolves by item identity, so reordering the mapping only reshuffles
# the play-UI machine palette, which renders in this order -- it has no
# functional effect. The jnp membership array is built in
# factoriax.placement, its sole engine consumer.
PLACEABLE_ITEM_LIST: tuple[int, ...] = tuple(int(it) for it in ITEM_TO_MACHINE)

RESOURCE_ITEM_LIST: tuple[int, ...] = tuple(
    i for i in range(1, NUM_ITEM_TYPES) if i not in PLACEABLE_ITEM_LIST
)

# ---------------------------------------------------------------------------
# Direction and movement
# ---------------------------------------------------------------------------


class Direction(IntEnum):
    """Compass facing directions for players and machines."""

    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


# (dx, dy) offset per compass direction, indexed by Direction value.
DIRECTIONS = jnp.array(
    [
        [0, 0],  # 0: NONE / invalid
        [-1, 0],  # 1: LEFT
        [1, 0],  # 2: RIGHT
        [0, -1],  # 3: UP
        [0, 1],  # 4: DOWN
    ],
    dtype=jnp.int32,
)

# ---------------------------------------------------------------------------
# Action categories (DEF2)
# ---------------------------------------------------------------------------
# The fixed (non-parametric) action categories. DEF2 composes the flat
# Action enum from these plus the parametric families generated from the
# item categories (Placement per Machine, Craft per non-resource item,
# Deposit per item). Like the item categories, their own integer values are
# inert -- the real action id lives on Action. A test
# (test_action_categories) guards that these cover today's fixed actions.


class MoveAction(IntEnum):
    """Movement and facing. Values inert (the action id is on Action)."""

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
    """Non-parametric world interactions. Values inert (see MoveAction)."""

    MINE = 0
    PICKUP = 1
    WITHDRAW = 2
    REPAIR = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    ROTATE_UP = 6
    ROTATE_DOWN = 7


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------


# The items each parametric action family addresses, in composition order.
# Placement covers every Machine, Craft every non-resource item (the design
# invariant: craftable <=> non-resource), Deposit every non-EMPTY item. The
# Action members and the game_logic action->item resolution tables both derive
# from these tuples, so the two cannot drift and no family can fall short of
# its item category the way hand-numbering left deposit (no LIMESTONE) and
# craft (no rocket parts) short.
PLACEMENT_ITEMS: tuple[Machine, ...] = tuple(m for m in Machine if m.name != "NONE")
CRAFT_ITEMS: tuple[HalfFabricate | Machine, ...] = tuple(
    m for cat in (HalfFabricate, Machine) for m in cat if m.name != "NONE"
)
DEPOSIT_ITEMS: tuple[Resource | HalfFabricate | Machine, ...] = tuple(
    m for cat in (Resource, HalfFabricate, Machine) for m in cat if m.name != "NONE"
)

# Flat member names of the composed Action enum, in family order: the two
# fixed families (MoveAction, InteractAction) then the three parametric
# families. Nothing hand-assigns a flat value; the categories are the source.
_ACTION_NAMES: list[str] = [
    *(a.name for a in MoveAction),
    *(a.name for a in InteractAction),
    *(f"PLACE_{m.name}" for m in PLACEMENT_ITEMS),
    *(f"CRAFT_{m.name}" for m in CRAFT_ITEMS),
    *(f"DEPOSIT_{m.name}" for m in DEPOSIT_ITEMS),
]

Action = IntEnum("Action", _ACTION_NAMES, start=0)
Action.__doc__ = """Player actions using compound action design.

Every action is self-contained: placement, deposit, and withdraw actions
name the specific item type so no slot cursor is needed. Movement actions
move in absolute map directions; FACE_* snaps facing without moving.

Composed from MoveAction + InteractAction (the fixed families) followed by
the parametric Placement/Craft/Deposit families generated from the item
categories, so the action layout tracks the item set automatically.
"""

# Base offsets for arithmetic dispatch of compound actions, derived from the
# fixed-family and parametric-family sizes -- no hand-tied member references.
PLACE_BASE: int = len(MoveAction) + len(InteractAction)
CRAFT_BASE: int = PLACE_BASE + len(PLACEMENT_ITEMS)
DEPOSIT_BASE: int = CRAFT_BASE + len(CRAFT_ITEMS)
ROTATE_BASE: int = len(MoveAction) + int(InteractAction.ROTATE_LEFT)

# The PLACE_* / ROTATE_* / CRAFT_* / DEPOSIT_* action-offset resolution tables
# live with the step dispatcher in factoriax.game_logic -- they are dispatch
# wiring, not environment constants.

# ---------------------------------------------------------------------------
# Block/terrain constants
# ---------------------------------------------------------------------------

MINEABLE_BLOCKS = jnp.array(
    [
        BlockType.COAL,
        BlockType.IRON,
        BlockType.COPPER,
        BlockType.TIN,
        BlockType.SILICON,
        BlockType.LIMESTONE,
    ],
)

SOLID_BLOCKS = jnp.array(
    [BlockType.WATER, BlockType.OUT_OF_BOUNDS],
    dtype=jnp.int32,
)

BLOCK_MAX_RESOURCES = 30000

NUM_ACTIONS = len(Action)
MAX_ACHIEVEMENTS = 64


# ---------------------------------------------------------------------------
# Machine slot roles
# ---------------------------------------------------------------------------


class SlotRole(IntEnum):
    """Slot roles for machine inventory display (editor compat)."""

    NONE = 0
    INPUT = 1
    OUTPUT = 2
    STORAGE = 3
    FUEL = 4
