"""Constants and enumerations for the FactoriaX environment."""

from enum import IntEnum

import jax.numpy as jnp
import numpy as np


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
    NEST = 9  # Temporary backward-compat (Stage 1)
    LIMESTONE = 10  # Refractory feedstock; pairs with COAL in furnace


class ItemType(IntEnum):
    """Item types that can be stored in inventory."""

    EMPTY = 0
    COAL = 1
    IRON_ORE = 2
    COPPER_ORE = 3
    TIN_ORE = 4
    SILICON = 5
    IRON_PLATE = 6
    COPPER_PLATE = 7
    TIN_PLATE = 8
    WAFER = 9
    FRAME = 10
    CIRCUIT = 11
    WIRE = 12
    MOTOR = 13
    SENSOR = 14
    CONVEYOR_BELT = 15
    MINER = 16
    ASSEMBLER = 17
    PALLET = 18
    ARM = 19
    BASIC_SCIENCE_PACK = 20
    ADVANCED_SCIENCE_PACK = 21
    ROCKET = 22
    FURNACE = 23
    REFRACTORY = 24
    HULL = 25
    ENGINE_UNIT = 26
    AVIONICS = 27
    ROCKET_CORE = 28
    SCIENCE_LAB = 29
    # LIMESTONE is appended at the end so existing ItemType ids and the
    # CRAFT_ACTION_TO_RECIPE slot ordering stay stable. Pairs with COAL
    # in the REFRACTORY recipe so every furnace recipe takes two inputs.
    LIMESTONE = 30
    # Belt-network pieces — the same logistical tier as CONVEYOR_BELT.
    # Appended at the end of ItemType so existing ids stay stable; the
    # corresponding CRAFT_SPLITTER / CRAFT_CROSSING actions extend the
    # CRAFT-addressable range (recipe indices 19, 20 — between
    # SCIENCE_LAB at 18 and the legacy machine-only REFRACTORY at 21).
    SPLITTER = 31
    CROSSING = 32


class MachineType(IntEnum):
    """Machine types that can be placed on tiles."""

    NONE = 0
    MINER = 1
    PALLET = 2
    ASSEMBLER = 3
    CONVEYOR_BELT = 4
    ARM = 5
    ROCKET = 6
    FURNACE = 7
    SCIENCE_LAB = 8
    # Belt-network pieces. SPLITTER reuses ent_buf (stack=2) and outputs
    # to the two perpendicular sides of its facing direction. CROSSING
    # reuses ent_asm_in[0..1] for two independent per-axis buffers so
    # the vertical and horizontal streams can co-exist on one tile
    # without mixing.
    SPLITTER = 9
    CROSSING = 10


NUM_ITEM_TYPES = len(ItemType)
MAX_STACK_SIZE = 64
MAX_MACHINE_STACK_SIZE = 64

# Canonical dtypes for state arrays.
INVENTORY_COUNT_DTYPE = jnp.int32
MACHINE_INVENTORY_COUNT_DTYPE = jnp.int16
BLOCK_RESOURCE_DTYPE = jnp.int16

# Max distinct item types a machine can hold simultaneously.
# SPLITTER holds one type (its stack-of-2 buffer); CROSSING holds two
# (one per axis slot in ent_asm_in[0..1]).
MACHINE_MAX_TYPES = jnp.array(
    [0, 2, 1, 4, 1, 1, 0, 2, 2, 1, 2],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET, FURNACE, SCIENCE_LAB,
    # SPLITTER, CROSSING
    dtype=jnp.int32,
)

# Max stack count per item type per machine type.
# SPLITTER buffer caps at 2 (the user-specified stack-of-2 splitter).
# CROSSING per-axis stack caps at 2: one cell of slack lets a fed
# crossing both drain to its output and accept from its input on the
# same tick, giving full one-tile-per-tick throughput in saturated
# chains without a separate look-ahead pass.
MACHINE_MAX_STACK = jnp.array(
    [0, MAX_MACHINE_STACK_SIZE, 256, 1000, 3, 1, 0, 1000, 1000, 2, 2],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET, FURNACE, SCIENCE_LAB,
    # SPLITTER, CROSSING
    dtype=jnp.int16,
)

# Max stack count per item type for the player inventory.
PLAYER_MAX_STACK = jnp.array(
    [
        0,  # EMPTY
        1024,  # COAL
        1024,  # IRON_ORE
        1024,  # COPPER_ORE
        1024,  # TIN_ORE
        1024,  # SILICON
        1024,  # IRON_PLATE
        1024,  # COPPER_PLATE
        1024,  # TIN_PLATE
        1024,  # WAFER
        1024,  # FRAME
        1024,  # CIRCUIT
        1024,  # WIRE
        1024,  # MOTOR
        1024,  # SENSOR
        128,  # CONVEYOR_BELT
        128,  # MINER
        128,  # ASSEMBLER
        128,  # PALLET
        128,  # ARM
        1024,  # BASIC_SCIENCE_PACK
        1024,  # ADVANCED_SCIENCE_PACK
        128,  # ROCKET
        128,  # FURNACE
        1024,  # REFRACTORY
        128,  # HULL
        128,  # ENGINE_UNIT
        128,  # AVIONICS
        128,  # ROCKET_CORE
        128,  # SCIENCE_LAB
        1024,  # LIMESTONE
        128,  # SPLITTER
        128,  # CROSSING
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

# Recipes are defined in factoriax.recipes (single source of truth).
from factoriax.recipes import (  # noqa: E402, F401
    BASE_RECIPES,
    CRAFT_ACTION_TO_RECIPE,
    MAX_RECIPE_INPUTS,
    NUM_RECIPES,
    OUTPUT_TO_RECIPE,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_NAMES,
    RECIPE_OUTPUT_COUNTS,
    RECIPE_OUTPUTS,
    RECIPE_TICKS,
)

# ---------------------------------------------------------------------------
# Science packs
# ---------------------------------------------------------------------------

# Tracked per-type by SCIENCE_LAB entities. The per-step delta vector
# in EnvState.science_consumed_step is sized to NUM_SCIENCE_PACK_TYPES
# and indexed by position in SCIENCE_PACK_TYPES.
NUM_SCIENCE_PACK_TYPES: int = 2
SCIENCE_PACK_TYPES: tuple[int, ...] = (
    int(ItemType.BASIC_SCIENCE_PACK),
    int(ItemType.ADVANCED_SCIENCE_PACK),
)

# Inverse lookup: ItemType -> index in SCIENCE_PACK_TYPES (0 or 1),
# or -1 for non-pack items. Used by lab consumption logic to bucket
# arbitrary slot contents into the delta vector.
SCIENCE_PACK_INDEX = (
    jnp.full(NUM_ITEM_TYPES, -1, dtype=jnp.int8)
    .at[ItemType.BASIC_SCIENCE_PACK]
    .set(0)
    .at[ItemType.ADVANCED_SCIENCE_PACK]
    .set(1)
)

# Whether an item type is a science pack that a lab will consume.
IS_SCIENCE_PACK = (
    jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.bool_)
    .at[ItemType.BASIC_SCIENCE_PACK]
    .set(True)
    .at[ItemType.ADVANCED_SCIENCE_PACK]
    .set(True)
)

# ---------------------------------------------------------------------------
# Placeable items and machine mappings
# ---------------------------------------------------------------------------

# The item<->machine bijection: every placeable item and the machine it
# becomes when placed. This is the anchor for the machine cluster -- the
# placeable set below and the jnp gather arrays in factoriax.placement are
# all projections of it.
ITEM_TO_MACHINE = {
    ItemType.MINER: MachineType.MINER,
    ItemType.PALLET: MachineType.PALLET,
    ItemType.CONVEYOR_BELT: MachineType.CONVEYOR_BELT,
    ItemType.ASSEMBLER: MachineType.ASSEMBLER,
    ItemType.ARM: MachineType.ARM,
    ItemType.ROCKET: MachineType.ROCKET,
    ItemType.FURNACE: MachineType.FURNACE,
    ItemType.SCIENCE_LAB: MachineType.SCIENCE_LAB,
    ItemType.SPLITTER: MachineType.SPLITTER,
    ItemType.CROSSING: MachineType.CROSSING,
}

# Items that place a machine when used -- exactly the keys of the mapping
# above, so the two cannot drift. Membership only: the PLACE_* action and
# UI palette order are interface concerns owned by their dispatchers, not
# this definition. The jnp membership array is built in factoriax.placement,
# its sole engine consumer.
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
# Action enum
# ---------------------------------------------------------------------------


class Action(IntEnum):
    """Player actions using compound action design.

    Every action is self-contained: placement, deposit, and withdraw
    actions name the specific item type so no slot cursor is needed.
    Movement actions move in absolute map directions. FACE_* snaps
    facing without moving.
    """

    # Movement (9)
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    FACE_UP = 5
    FACE_DOWN = 6
    FACE_LEFT = 7
    FACE_RIGHT = 8

    # World (2)
    MINE = 9
    PICKUP = 10

    # Placement — one per placeable machine type (10)
    PLACE_MINER = 11
    PLACE_PALLET = 12
    PLACE_BELT = 13
    PLACE_ASSEMBLER = 14
    PLACE_ARM = 15
    PLACE_ROCKET = 16
    PLACE_FURNACE = 17
    PLACE_SCIENCE_LAB = 18
    PLACE_SPLITTER = 19
    PLACE_CROSSING = 20

    # Crafting — one per recipe output (21)
    CRAFT_IRON_PLATE = 21
    CRAFT_COPPER_PLATE = 22
    CRAFT_TIN_PLATE = 23
    CRAFT_WAFER = 24
    CRAFT_FRAME = 25
    CRAFT_CIRCUIT = 26
    CRAFT_WIRE = 27
    CRAFT_MOTOR = 28
    CRAFT_SENSOR = 29
    CRAFT_BELT = 30
    CRAFT_MINER = 31
    CRAFT_ASSEMBLER = 32
    CRAFT_PALLET = 33
    CRAFT_ARM = 34
    CRAFT_FURNACE = 35
    CRAFT_BASIC_SCIENCE = 36
    CRAFT_ADV_SCIENCE = 37
    CRAFT_ROCKET = 38
    CRAFT_SCIENCE_LAB = 39
    CRAFT_SPLITTER = 40
    CRAFT_CROSSING = 41

    # Deposit — one per non-EMPTY item type (31)
    DEPOSIT_COAL = 42
    DEPOSIT_IRON_ORE = 43
    DEPOSIT_COPPER_ORE = 44
    DEPOSIT_TIN_ORE = 45
    DEPOSIT_SILICON = 46
    DEPOSIT_IRON_PLATE = 47
    DEPOSIT_COPPER_PLATE = 48
    DEPOSIT_TIN_PLATE = 49
    DEPOSIT_WAFER = 50
    DEPOSIT_FRAME = 51
    DEPOSIT_CIRCUIT = 52
    DEPOSIT_WIRE = 53
    DEPOSIT_MOTOR = 54
    DEPOSIT_SENSOR = 55
    DEPOSIT_BELT = 56
    DEPOSIT_MINER = 57
    DEPOSIT_ASSEMBLER = 58
    DEPOSIT_PALLET = 59
    DEPOSIT_ARM = 60
    DEPOSIT_BASIC_SCIENCE = 61
    DEPOSIT_ADV_SCIENCE = 62
    DEPOSIT_ROCKET = 63
    DEPOSIT_FURNACE = 64
    DEPOSIT_REFRACTORY = 65
    DEPOSIT_HULL = 66
    DEPOSIT_ENGINE_UNIT = 67
    DEPOSIT_AVIONICS = 68
    DEPOSIT_ROCKET_CORE = 69
    DEPOSIT_SCIENCE_LAB = 70
    DEPOSIT_SPLITTER = 71
    DEPOSIT_CROSSING = 72

    # Withdraw — single action; machines have one output slot so no
    # per-item selection is needed (mirrors PICKUP / MINE).
    WITHDRAW = 73

    # Machine rotation — absolute direction set (4)
    ROTATE_LEFT = 74
    ROTATE_RIGHT = 75
    ROTATE_UP = 76
    ROTATE_DOWN = 77

    # Repair the machine in front of the player. The base engine
    # restores the target's health to its configured maximum;
    # wrappers override repair semantics by pre-empting this action.
    REPAIR = 78


# Base offsets for arithmetic dispatch of compound actions.
PLACE_BASE: int = Action.PLACE_MINER
CRAFT_BASE: int = Action.CRAFT_IRON_PLATE
DEPOSIT_BASE: int = Action.DEPOSIT_COAL
ROTATE_BASE: int = Action.ROTATE_LEFT

# The PLACE_* / ROTATE_* / CRAFT_* action-offset resolution tables live with
# the step dispatcher in factoriax.game_logic — they are dispatch wiring, not
# environment constants.

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
# Temporary backward-compat constants (will be removed in Stage 2+)
# ---------------------------------------------------------------------------

#: Default maximum health for every placeable machine. Wrappers can
#: tune per-type values via :class:`~factoriax.machine_config.MachineConfig`.
MAX_HEALTH: int = 256

NUM_INVENTORY_SLOTS: int = 10
MAX_MACHINE_INVENTORY_SLOTS: int = 8

TURN_RIGHT_MAP = jnp.array([0, 3, 4, 2, 1], dtype=jnp.int32)

# Per-MachineType slot counts (indexed by MachineType value).
# SCIENCE_LAB: 2 input slots (one per pack type), no output.
# SPLITTER: 1 buffer slot (the stack-of-2 ent_buf).
# CROSSING: 2 axis slots (vertical = ent_asm_in[0], horizontal = ent_asm_in[1]).
MACHINE_NUM_SLOTS = np.array([0, 1, 1, 3, 1, 0, 0, 3, 2, 1, 2], dtype=np.int32)


class SlotRole(IntEnum):
    """Slot roles for machine inventory display (editor compat)."""

    NONE = 0
    INPUT = 1
    OUTPUT = 2
    STORAGE = 3
    FUEL = 4


MACHINE_SLOT_ROLES = np.array(
    [
        [SlotRole.NONE] * 8,  # NONE
        [SlotRole.OUTPUT] + [SlotRole.NONE] * 7,  # MINER
        [SlotRole.STORAGE] * 8,  # PALLET
        [SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT] + [SlotRole.NONE] * 5,
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,  # CONVEYOR_BELT
        [SlotRole.NONE] * 8,  # ARM (instant, no buffer)
        [SlotRole.NONE] * 8,  # ROCKET
        [SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT] + [SlotRole.NONE] * 5,
        [SlotRole.INPUT, SlotRole.INPUT] + [SlotRole.NONE] * 6,  # SCIENCE_LAB
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,  # SPLITTER (stack-of-2 buf)
        [SlotRole.STORAGE, SlotRole.STORAGE] + [SlotRole.NONE] * 6,  # CROSSING
    ],
    dtype=np.int32,
)
