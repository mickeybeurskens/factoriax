"""Constants and enumerations for the FactoriaX environment.

Pure-Python definitions only. Their derived JAX arrays (gather tables, the
state-array dtypes) live in :mod:`factoriax.engine.tables`.

Enum values here are a wire format. :func:`factoriax.engine.levels.save_level`
writes raw integers for terrain and machine kinds into level JSON, observations
expose them to trained policies, and :class:`Action` values are the policy
output space. Renumbering an existing member invalidates saved levels and
trained checkpoints, so new members are appended rather than inserted.
"""

from enum import IntEnum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BlockType(IntEnum):
    """Terrain kind of one tile, stored in ``EnvState.map``.

    A tile carries exactly one block type. ``INVALID = 0`` is a sentinel: the
    editor palette omits it, so a tile holding it means the map array was left
    zero-filled. The engine gives it no behavior of its own and treats it as
    walkable, non-mineable ground. ``OUT_OF_BOUNDS`` never occupies a stored
    tile either: queries outside the map return it, and observation padding
    fills the border with it so a policy can see the map edge.

    ``WATER`` and ``OUT_OF_BOUNDS`` are the terrain that blocks movement,
    listed in ``factoriax.engine.tables.SOLID_BLOCKS``. ``DIRT`` is plain
    walkable ground. Terrain is not the whole rule: a placed machine blocks
    its own tile too, every kind except ``CONVEYOR_BELT``.

    The remaining six are ore tiles. They are the keys of
    :data:`BLOCK_TO_ITEM` and the only mineable tiles.
    """

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
    """Facing of a player or machine, in absolute map directions.

    Grid coordinates run right and down from the top-left tile, so ``UP`` is
    the ``-y`` (decreasing row) direction and ``DOWN`` is ``+y``. Facing is
    absolute, never relative to the entity's own heading.

    Numbering starts at 1. Zero is not a member, but code that stores a
    direction uses it as "no direction": ``factoriax.engine.tables.DIRECTIONS``
    reserves index 0 for a zero offset, and :class:`~factoriax.engine.levels.Level`
    leaves unset machine directions at 0.
    """

    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class SlotRole(IntEnum):
    """Purpose of one inventory slot on a machine.

    Declared per machine kind by
    ``factoriax.playground.editor.slot_display.MACHINE_SLOTS``. The role labels
    the slot for the editor and the play UI. The simulation stores machine
    contents in its own entity arrays and never reads a role.

    ``NONE`` also pads
    ``factoriax.playground.editor.slot_display.MACHINE_SLOT_ROLES`` out to the
    widest machine, so it marks a slot that does not exist. No shipped machine
    declares a ``FUEL`` slot; the editor still handles the role, which
    restricts the slot to coal.
    """

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
    """Raw material obtained by mining an ore tile.

    One member per ore :class:`BlockType`, paired by :data:`BLOCK_TO_ITEM`.
    Resources are recipe inputs only: nothing crafts them and nothing places
    them.
    """

    NONE = 0
    COAL = 1
    IRON_ORE = 2
    COPPER_ORE = 3
    TIN_ORE = 4
    SILICON = 5
    LIMESTONE = 6


class HalfFabricate(IntEnum):
    """Crafted item that cannot be placed on the map.

    Every member is a recipe output. Members are ordered roughly by tech
    depth, from smelted plates up to rocket components, but the order carries
    no meaning to the engine.
    """

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
    TIER1_SCIENCE_PACK = 10
    TIER2_SCIENCE_PACK = 11
    REFRACTORY = 12
    HULL = 13
    ENGINE_UNIT = 14
    AVIONICS = 15
    ROCKET_CORE = 16
    # Appended last so existing HalfFabricate values stay stable; the
    # composed ItemType still renumbers the Machine-derived items (+1)
    # and Action grows its CRAFT_/DEPOSIT_ entries (+2).
    TIER3_SCIENCE_PACK = 17


class Machine(IntEnum):
    """Placeable machine kind, and the entity tag the engine stores per tile.

    A machine is both an item (carried in inventory, crafted) and an entity
    (placed on the map), so this enum does double duty. Its member names
    compose :class:`ItemType` and the ``PLACE_`` / ``CRAFT_`` / ``DEPOSIT_``
    action families; its values are what ``EnvState.machine_types`` and
    ``EnvState.ent_type`` hold. ``NONE = 0`` means no machine on the tile, the
    analog of ``ItemType.EMPTY = 0``.

    Values index the per-machine rows of
    ``factoriax.engine.tables.MACHINE_MAX_STACK`` and of the editor's
    ``factoriax.playground.editor.slot_display.MACHINE_SLOT_ROLES``. Both are
    built by iterating this enum over a dict keyed by member, so adding a
    member without an entry in each raises ``KeyError`` at import.
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
ItemType.__doc__ = """Anything that can occupy an inventory slot.

Composed at import time from the member names of :class:`Resource`,
:class:`HalfFabricate`, and :class:`Machine`, in that order, with the ``NONE``
placeholders dropped and ``EMPTY = 0`` prepended. ``EMPTY`` marks an unused
slot, so a slot is empty by type, not by a zero count.

Values are positional within that concatenation. Appending to an earlier
category therefore shifts every later item, which changes the item axis of
``EnvState.player_inventory`` and of the observation. Appending to
:class:`Machine`, the last category, leaves every existing item value
unchanged.
"""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class MoveAction(IntEnum):
    """Action that moves the player or turns it in place.

    The first block of :class:`Action`, at the same values. Directions are
    absolute map directions, not relative to the current facing. ``UP`` and the
    three other moves step one tile and also set facing; ``FACE_*`` sets facing
    without moving. A move into a solid tile leaves the position unchanged and
    still updates facing.
    """

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
    """Action on the tile the player faces, taking no item parameter.

    The second block of :class:`Action`, offset by ``len(MoveAction)``. Every
    member acts on the single tile in front of the player, so the target is
    chosen by facing rather than by a coordinate argument. ``ROTATE_*`` sets a
    placed machine to an absolute :class:`Direction` and must stay contiguous
    and in ``LEFT, RIGHT, UP, DOWN`` order, because the dispatcher reads
    ``action - ROTATE_BASE`` as an index.

    Members with no valid target are a no-op rather than an error: the step
    consumes a timestep and leaves the state unchanged.
    """

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
# the actions.py offset->item tables both derive from these, so an action's
# offset within its block is the item's index in the matching tuple.

#: Items with a ``PLACE_`` action, one per placeable machine kind.
PLACEMENT_ITEMS: tuple[Machine, ...] = tuple(m for m in Machine if m.name != "NONE")
#: Items with a ``CRAFT_`` action: the half-fabricates and the machines. Raw
#: resources are mined rather than crafted, and ``EMPTY`` is not an item.
CRAFT_ITEMS: tuple[HalfFabricate | Machine, ...] = tuple(
    m for cat in (HalfFabricate, Machine) for m in cat if m.name != "NONE"
)
#: Items with a ``DEPOSIT_`` action: every item type except ``EMPTY``.
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
Action.__doc__ = """Action a player issues per step.

Composed at import time from five contiguous blocks, in order:
:class:`MoveAction`, :class:`InteractAction`, then ``PLACE_``, ``CRAFT_``, and
``DEPOSIT_`` over :data:`PLACEMENT_ITEMS`, :data:`CRAFT_ITEMS`, and
:data:`DEPOSIT_ITEMS`. :data:`PLACE_BASE`, :data:`CRAFT_BASE`, and
:data:`DEPOSIT_BASE` give the first value of each parametric block.

Actions are compound: the item is part of the action name, so an agent needs no
separate slot cursor or item cursor and the action space stays flat. ``WITHDRAW``
is the exception, since a machine exposes one output slot at a time and needs no
item argument.

This is the policy action space. ``NUM_ACTIONS`` is its size, and the integer
values are what a trained checkpoint has learned, so inserting a member
invalidates existing checkpoints.
"""


# ---------------------------------------------------------------------------
# Derived sizes and offsets
# ---------------------------------------------------------------------------

#: Width of the item axis of every inventory array, ``EMPTY`` included.
NUM_ITEM_TYPES = len(ItemType)
#: Size of the discrete action space. Valid actions are ``0`` to
#: ``NUM_ACTIONS - 1``.
NUM_ACTIONS = len(Action)
#: Fixed width of the ``EnvState.achievements_unlocked`` bit vector. A scenario
#: may define fewer achievements; the unused trailing bits stay False. Raising
#: this widens the observation and invalidates trained checkpoints.
MAX_ACHIEVEMENTS = 64
#: Units of ore an ore tile can hold when a level does not set a count. Also the
#: divisor that normalizes ``block_resources`` into the observation, so a tile
#: above this value normalizes past 1.0. ``EnvState.block_resources`` is int16,
#: so this must stay under 32767.
BLOCK_MAX_RESOURCES = 30000

# Base offsets for the parametric action families. The PLACE_/CRAFT_/DEPOSIT_
# offset->item tables live with the dispatcher in factoriax.engine.actions.

#: First ``PLACE_`` action value.
PLACE_BASE: int = len(MoveAction) + len(InteractAction)
#: First ``CRAFT_`` action value.
CRAFT_BASE: int = PLACE_BASE + len(PLACEMENT_ITEMS)
#: First ``DEPOSIT_`` action value.
DEPOSIT_BASE: int = CRAFT_BASE + len(CRAFT_ITEMS)
#: First ``ROTATE_`` action value. ``action - ROTATE_BASE`` is an index in
#: ``0..3`` into ``LEFT, RIGHT, UP, DOWN``, one less than the
#: :class:`Direction` value.
ROTATE_BASE: int = len(MoveAction) + int(InteractAction.ROTATE_LEFT)


# ---------------------------------------------------------------------------
# Item / machine / block mappings
# ---------------------------------------------------------------------------

#: Machine each placeable item becomes once placed. An item absent here cannot
#: be placed. The pairing is one to one, so :mod:`factoriax.engine.tables`
#: builds both the forward gather array and its inverse from this one dict and
#: the two cannot disagree.
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

#: Placeable item ids, the keys of :data:`ITEM_TO_MACHINE`. Order is incidental:
#: it sets the play-UI palette order, and the agent resolves placement by item
#: identity instead.
PLACEABLE_ITEM_LIST: tuple[int, ...] = tuple(int(it) for it in ITEM_TO_MACHINE)

#: Item ids that are not placeable, meaning resources and intermediates.
#: The complement of :data:`PLACEABLE_ITEM_LIST` over the item range, with
#: ``EMPTY`` excluded. Used by the play UI to split the inventory display.
RESOURCE_ITEM_LIST: tuple[int, ...] = tuple(
    i for i in range(1, NUM_ITEM_TYPES) if i not in PLACEABLE_ITEM_LIST
)

#: Item each ore block yields when mined. The keys are the mineable blocks;
#: any block absent here yields nothing.
BLOCK_TO_ITEM: dict[int, int] = {
    BlockType.COAL: ItemType.COAL,
    BlockType.IRON: ItemType.IRON_ORE,
    BlockType.COPPER: ItemType.COPPER_ORE,
    BlockType.TIN: ItemType.TIN_ORE,
    BlockType.SILICON: ItemType.SILICON,
    BlockType.LIMESTONE: ItemType.LIMESTONE,
}

#: Science pack item ids that a ``SCIENCE_LAB`` consumes, in tier order.
#: A pack's position here is its index into the per-step delta vector
#: ``EnvState.science_consumed_step``, so the order is part of the state
#: layout that reward functions and analysis code read.
SCIENCE_PACK_TYPES: tuple[int, ...] = (
    int(ItemType.TIER1_SCIENCE_PACK),
    int(ItemType.TIER2_SCIENCE_PACK),
    int(ItemType.TIER3_SCIENCE_PACK),
)
#: Length of ``EnvState.science_consumed_step``.
NUM_SCIENCE_PACK_TYPES: int = len(SCIENCE_PACK_TYPES)
