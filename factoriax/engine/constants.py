"""Constants and enumerations for the FactoriaX environment.

This module holds pure-Python definitions only. The JAX arrays that come from
them (the gather tables and the state-array dtypes) live in
:mod:`factoriax.engine.tables`.

The enum values here are a wire format.
:func:`factoriax.engine.levels.save_level` writes raw integers for terrain and
machine kinds into level JSON. The observation exposes the same integers to
trained policies, and the :class:`Action` values are the output space of a
policy. A new number for an existing member therefore invalidates saved levels
and trained checkpoints. New members must go at the end of an enum, never in
the middle.
"""

from enum import IntEnum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BlockType(IntEnum):
    """Terrain kind of one tile, stored in ``EnvState.map``.

    A tile carries exactly one block type. ``INVALID = 0`` is a sentinel. The
    editor palette does not offer it, so a tile with this value shows that the
    map array is still full of zeros. The engine gives it no behavior of its
    own and treats it as walkable ground that no player can mine.
    ``OUT_OF_BOUNDS`` also never occupies a stored tile. A query outside the
    map returns this value, and the observation padding fills the border with
    it. A policy can therefore see the edge of the map.

    ``WATER`` and ``OUT_OF_BOUNDS`` are the terrain that blocks movement.
    ``factoriax.engine.tables.SOLID_BLOCKS`` lists them. ``DIRT`` is plain
    walkable ground. Terrain is not the whole rule. A placed machine also
    blocks its own tile, for every machine kind except ``CONVEYOR_BELT``.

    The other six members are ore tiles. They are the keys of
    :data:`BLOCK_TO_ITEM` and the only tiles that a player can mine.
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

    Grid coordinates run right and down from the top-left tile. ``UP`` is
    therefore the ``-y`` direction, which moves to a lower row number, and
    ``DOWN`` is ``+y``. A facing is absolute. It is never relative to the
    direction that the entity itself points to.

    The values start at 1. Zero is not a member, but code that stores a
    direction uses zero for "no direction".
    ``factoriax.engine.tables.DIRECTIONS`` keeps index 0 for a zero offset, and
    :class:`~factoriax.engine.levels.Level` leaves a machine direction at 0
    until something sets it.
    """

    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class SlotRole(IntEnum):
    """Purpose of one inventory slot on a machine.

    ``factoriax.playground.editor.slot_display.MACHINE_SLOTS`` declares the
    roles for each machine kind. The role labels the slot for the editor and
    the play UI. The simulation stores the contents of a machine in its own
    entity arrays and never reads a role.

    ``NONE`` also fills
    ``factoriax.playground.editor.slot_display.MACHINE_SLOT_ROLES`` to the
    width of the widest machine, so it marks a slot that does not exist. No
    machine in this repository declares a ``FUEL`` slot. The editor still
    supports the role, which limits the slot to coal.
    """

    NONE = 0
    INPUT = 1
    OUTPUT = 2
    STORAGE = 3
    FUEL = 4


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

# The three item categories. Each holds a NONE=0 placeholder that every
# derivative drops. ItemType comes from their member names.


class Resource(IntEnum):
    """Raw material that a player gets from an ore tile.

    There is one member for each ore :class:`BlockType`, and
    :data:`BLOCK_TO_ITEM` holds the pairs. A resource is a recipe input only.
    No recipe makes one, and no action places one on the map.
    """

    NONE = 0
    COAL = 1
    IRON_ORE = 2
    COPPER_ORE = 3
    TIN_ORE = 4
    SILICON = 5
    LIMESTONE = 6


class HalfFabricate(IntEnum):
    """Crafted item that a player cannot place on the map.

    Every member is a recipe output. The members follow tech depth
    approximately, from smelted plates to rocket parts. The engine gives this
    order no meaning.
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
    # Last in the list, so the existing HalfFabricate values stay stable.
    # ItemType still renumbers the Machine items (+1), and Action gains
    # CRAFT_/DEPOSIT_ entries (+2).
    TIER3_SCIENCE_PACK = 17


class Machine(IntEnum):
    """Placeable machine kind, and the entity tag that the engine stores per tile.

    A machine is both an item and an entity. As an item, a player crafts it and
    carries it in the inventory. As an entity, it stands on a map tile. The
    member names compose :class:`ItemType` and the ``PLACE_``, ``CRAFT_``, and
    ``DEPOSIT_`` actions. The member values are what ``EnvState.machine_types``
    and ``EnvState.ent_type`` hold. ``NONE = 0`` means no machine on the tile,
    the analog of ``ItemType.EMPTY = 0``.

    The values index the per-machine rows of
    ``factoriax.engine.tables.MACHINE_MAX_STACK`` and of
    ``factoriax.playground.editor.slot_display.MACHINE_SLOT_ROLES`` in the
    editor. Both tables walk this enum over a dict that has one key for each
    member. A new member with no entry in each dict therefore raises
    ``KeyError`` at import.
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

This enum comes from the member names of :class:`Resource`,
:class:`HalfFabricate`, and :class:`Machine`, in that order. The ``NONE``
placeholders drop out, and ``EMPTY = 0`` comes first. ``EMPTY`` marks an unused
slot, so a slot is empty by type, not by a zero count.

Each value is the position of its name in that concatenation. A new member in
an earlier category therefore moves every later item to a new value. This
changes the item axis of ``EnvState.player_inventory`` and of the observation.
:class:`Machine` is the last category, so a new member there leaves every
existing item value unchanged.
"""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class MoveAction(IntEnum):
    """Action that moves the player or turns it in place.

    This is the first block of :class:`Action`, at the same values. The
    directions are absolute map directions. They are not relative to the
    current facing. ``UP`` and the three other moves step one tile and also set
    the facing. The ``FACE_*`` actions set the facing only, with no step. A
    move into a solid tile leaves the position unchanged and still sets the
    facing.
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
    """Action on the tile in front of the player, with no item parameter.

    This is the second block of :class:`Action`, at an offset of
    ``len(MoveAction)``. Every member acts on the one tile in front of the
    player. The facing selects the target, not a coordinate argument.
    ``ROTATE_*`` sets a placed machine to an absolute :class:`Direction`. These
    four members must stay next to each other, in ``LEFT, RIGHT, UP, DOWN``
    order, because the dispatcher reads ``action - ROTATE_BASE`` as an index.

    A member with no valid target does nothing, and this is not an error. The
    step uses one timestep and leaves the state unchanged.
    """

    MINE = 0
    PICKUP = 1
    WITHDRAW = 2
    REPAIR = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    ROTATE_UP = 6
    ROTATE_DOWN = 7


# Items that each parametric action block addresses, with NONE removed. Action
# and the offset->item tables in actions.py both come from these tuples, so the
# offset of an action inside its block is the index of its item here.

#: Items with a ``PLACE_`` action, one for each placeable machine kind.
PLACEMENT_ITEMS: tuple[Machine, ...] = tuple(m for m in Machine if m.name != "NONE")
#: Items with a ``CRAFT_`` action: the half-fabricates and the machines. A
#: player mines a raw resource and never crafts one, and ``EMPTY`` is not an
#: item.
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

This enum comes from five blocks that follow each other, in this order:
:class:`MoveAction`, :class:`InteractAction`, then ``PLACE_`` over
:data:`PLACEMENT_ITEMS`, ``CRAFT_`` over :data:`CRAFT_ITEMS`, and ``DEPOSIT_``
over :data:`DEPOSIT_ITEMS`. :data:`PLACE_BASE`, :data:`CRAFT_BASE`, and
:data:`DEPOSIT_BASE` give the first value of each parametric block.

The actions are compound. The item is part of the action name, so an agent
needs no separate slot cursor and no item cursor. The action space stays flat.
``WITHDRAW`` is the exception, because a machine exposes one output slot at a
time and needs no item argument.

This is the action space of a policy. ``NUM_ACTIONS`` is its size. A trained
checkpoint learned the integer values, so a new member in the middle
invalidates every existing checkpoint.
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
#: can define fewer achievements, and the unused bits at the end stay False. A
#: larger value widens the observation and invalidates trained checkpoints.
MAX_ACHIEVEMENTS = 64
#: Units of ore that an ore tile holds when a level sets no count. This value is
#: also the divisor that normalizes ``block_resources`` for the observation, so
#: a tile with more ore than this value normalizes to more than 1.0.
#: ``EnvState.block_resources`` is int16, so this value must stay less than
#: 32767.
BLOCK_MAX_RESOURCES = 30000

# Base offsets for the parametric action blocks. The PLACE_/CRAFT_/DEPOSIT_
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

#: Machine that each placeable item becomes on the map. An item that is absent
#: here has no ``PLACE_`` action. The pairs are one to one, so
#: :mod:`factoriax.engine.tables` builds both the forward gather array and its
#: inverse from this one dict. The two arrays therefore cannot disagree.
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

#: Placeable item ids, the keys of :data:`ITEM_TO_MACHINE`. The order sets the
#: palette order in the play UI and has no other meaning. An agent selects a
#: placement by item identity.
PLACEABLE_ITEM_LIST: tuple[int, ...] = tuple(int(it) for it in ITEM_TO_MACHINE)

#: Item ids that no ``PLACE_`` action can place: the resources and the
#: half-fabricates. This tuple is the complement of
#: :data:`PLACEABLE_ITEM_LIST` over the item range, without ``EMPTY``. The play
#: UI reads it to divide the inventory display.
RESOURCE_ITEM_LIST: tuple[int, ...] = tuple(
    i for i in range(1, NUM_ITEM_TYPES) if i not in PLACEABLE_ITEM_LIST
)

#: Item that each ore block gives to the player who mines it. The keys are the
#: blocks that a player can mine. A block that is absent here gives nothing.
BLOCK_TO_ITEM: dict[int, int] = {
    BlockType.COAL: ItemType.COAL,
    BlockType.IRON: ItemType.IRON_ORE,
    BlockType.COPPER: ItemType.COPPER_ORE,
    BlockType.TIN: ItemType.TIN_ORE,
    BlockType.SILICON: ItemType.SILICON,
    BlockType.LIMESTONE: ItemType.LIMESTONE,
}

#: Science pack item ids that a ``SCIENCE_LAB`` consumes, in tier order. The
#: position of a pack here is its index into the per-step delta vector
#: ``EnvState.science_consumed_step``. The order is therefore part of the state
#: layout, and reward functions and analysis code read it.
SCIENCE_PACK_TYPES: tuple[int, ...] = (
    int(ItemType.TIER1_SCIENCE_PACK),
    int(ItemType.TIER2_SCIENCE_PACK),
    int(ItemType.TIER3_SCIENCE_PACK),
)
#: Length of ``EnvState.science_consumed_step``.
NUM_SCIENCE_PACK_TYPES: int = len(SCIENCE_PACK_TYPES)
