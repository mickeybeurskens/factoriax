"""Constants and enumerations for the FactoriaX environment."""

from enum import IntEnum
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ASSETS_PATH = Path(__file__).parent / "assets"


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
    STEEL = 10
    CIRCUIT = 11
    WIRE = 12
    MOTOR = 13
    SENSOR = 14
    CONVEYOR_BELT = 15
    MINER = 16
    ASSEMBLER = 17
    PALLET = 18
    BASIC_SCIENCE_PACK = 19
    ADVANCED_SCIENCE_PACK = 20
    ROCKET = 21


class MachineType(IntEnum):
    """Machine types that can be placed on tiles."""

    NONE = 0
    MINER = 1
    PALLET = 2
    ASSEMBLER = 3
    CONVEYOR_BELT = 4
    ROCKET = 5


NUM_ITEM_TYPES = len(ItemType)
MAX_STACK_SIZE = 64
MAX_MACHINE_STACK_SIZE = 64

# Canonical dtypes for state arrays.
INVENTORY_COUNT_DTYPE = jnp.int32
MACHINE_INVENTORY_COUNT_DTYPE = jnp.int16
BLOCK_RESOURCE_DTYPE = jnp.int16

# Max distinct item types a machine can hold simultaneously.
MACHINE_MAX_TYPES = jnp.array(
    [0, 2, 1, 4, 1, 0],
    # NONE, MINER, PALLET, ASM, BELT, ROCKET
    dtype=jnp.int32,
)

# Max stack count per item type per machine type.
MACHINE_MAX_STACK = jnp.array(
    [0, MAX_MACHINE_STACK_SIZE, 256, 1000, MAX_MACHINE_STACK_SIZE, 0],
    # NONE, MINER, PALLET, ASM, BELT, ROCKET
    dtype=jnp.int32,
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
        1024,  # STEEL
        1024,  # CIRCUIT
        1024,  # WIRE
        1024,  # MOTOR
        1024,  # SENSOR
        128,  # CONVEYOR_BELT
        128,  # MINER
        128,  # ASSEMBLER
        128,  # PALLET
        1024,  # BASIC_SCIENCE_PACK
        1024,  # ADVANCED_SCIENCE_PACK
        128,  # ROCKET
    ],
    dtype=jnp.int32,
)

BLOCK_TO_ITEM: dict[int, int] = {
    BlockType.COAL: ItemType.COAL,
    BlockType.IRON: ItemType.IRON_ORE,
    BlockType.COPPER: ItemType.COPPER_ORE,
    BlockType.TIN: ItemType.TIN_ORE,
    BlockType.SILICON: ItemType.SILICON,
}

ITEM_COLORS: dict[int, tuple[int, int, int]] = {
    ItemType.COAL: (54, 54, 54),
    ItemType.IRON_ORE: (160, 140, 130),
    ItemType.COPPER_ORE: (170, 100, 50),
    ItemType.TIN_ORE: (180, 180, 170),
    ItemType.SILICON: (100, 110, 130),
    ItemType.IRON_PLATE: (192, 192, 192),
    ItemType.COPPER_PLATE: (184, 115, 51),
    ItemType.TIN_PLATE: (200, 200, 190),
    ItemType.WAFER: (80, 90, 140),
    ItemType.STEEL: (140, 150, 165),
    ItemType.CIRCUIT: (40, 160, 80),
    ItemType.WIRE: (200, 140, 60),
    ItemType.MOTOR: (100, 100, 180),
    ItemType.SENSOR: (180, 80, 80),
    ItemType.CONVEYOR_BELT: (220, 180, 50),
    ItemType.MINER: (0, 200, 0),
    ItemType.ASSEMBLER: (160, 80, 200),
    ItemType.PALLET: (170, 170, 175),
    ItemType.BASIC_SCIENCE_PACK: (200, 50, 50),
    ItemType.ADVANCED_SCIENCE_PACK: (50, 50, 200),
    ItemType.ROCKET: (240, 240, 240),
}

# Human-readable display names for each MachineType.
MACHINE_TYPE_NAMES: dict[int, str] = {
    int(MachineType.NONE): "None",
    int(MachineType.MINER): "Miner",
    int(MachineType.PALLET): "Pallet",
    int(MachineType.ASSEMBLER): "Assembler",
    int(MachineType.CONVEYOR_BELT): "Conveyor Belt",
    int(MachineType.ROCKET): "Rocket",
}

# Recipes are defined in factoriax.recipes (single source of truth).
from factoriax.recipes import (  # noqa: E402, F401
    ASSEMBLER_RECIPE_INPUT_COUNTS,
    ASSEMBLER_RECIPE_INPUT_ITEMS,
    ASSEMBLER_RECIPE_NAMES,
    ASSEMBLER_RECIPE_OUTPUTS,
    ASSEMBLER_RECIPE_TICKS,
    ASSEMBLER_RECIPES,
    CRAFT_ACTION_TO_RECIPE,
    MAX_ASSEMBLER_RECIPE_INPUTS,
    MAX_RECIPE_INPUTS,
    NUM_ASSEMBLER_RECIPES,
    NUM_RECIPES,
    OUTPUT_TO_RECIPE,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_NAMES,
    RECIPE_OUTPUTS,
    RECIPE_TICKS,
    RECIPES,
)

# ---------------------------------------------------------------------------
# Technology / research tree
# ---------------------------------------------------------------------------

NUM_TECHNOLOGIES: int = 2
RESEARCH_COST: int = 10  # science packs per unlock

# Maps science pack item type -> technology index.
SCIENCE_PACK_TO_TECH = (
    jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
    .at[ItemType.BASIC_SCIENCE_PACK]
    .set(0)
    .at[ItemType.ADVANCED_SCIENCE_PACK]
    .set(1)
)

# Whether an item type is a science pack that can be used for research.
IS_RESEARCH_ITEM = (
    jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.bool_)
    .at[ItemType.BASIC_SCIENCE_PACK]
    .set(True)
    .at[ItemType.ADVANCED_SCIENCE_PACK]
    .set(True)
)

# ---------------------------------------------------------------------------
# Placeable items and machine mappings
# ---------------------------------------------------------------------------

PLACEABLE_ITEMS = jnp.array(
    [
        ItemType.MINER,
        ItemType.PALLET,
        ItemType.CONVEYOR_BELT,
        ItemType.ASSEMBLER,
        ItemType.ROCKET,
    ],
    dtype=jnp.int32,
)

PLACEABLE_ITEM_LIST: tuple[int, ...] = (
    int(ItemType.MINER),
    int(ItemType.PALLET),
    int(ItemType.CONVEYOR_BELT),
    int(ItemType.ASSEMBLER),
    int(ItemType.ROCKET),
)

PLACEABLE_ITEM_SET: frozenset[int] = frozenset(PLACEABLE_ITEM_LIST)

RESOURCE_ITEM_LIST: tuple[int, ...] = tuple(
    i for i in range(1, NUM_ITEM_TYPES) if i not in PLACEABLE_ITEM_SET
)

ITEM_TO_MACHINE = {
    ItemType.MINER: MachineType.MINER,
    ItemType.PALLET: MachineType.PALLET,
    ItemType.CONVEYOR_BELT: MachineType.CONVEYOR_BELT,
    ItemType.ASSEMBLER: MachineType.ASSEMBLER,
    ItemType.ROCKET: MachineType.ROCKET,
}

ITEM_TO_MACHINE_ARRAY = jnp.array(
    [
        MachineType.NONE,  # EMPTY
        MachineType.NONE,  # COAL
        MachineType.NONE,  # IRON_ORE
        MachineType.NONE,  # COPPER_ORE
        MachineType.NONE,  # TIN_ORE
        MachineType.NONE,  # SILICON
        MachineType.NONE,  # IRON_PLATE
        MachineType.NONE,  # COPPER_PLATE
        MachineType.NONE,  # TIN_PLATE
        MachineType.NONE,  # WAFER
        MachineType.NONE,  # STEEL
        MachineType.NONE,  # CIRCUIT
        MachineType.NONE,  # WIRE
        MachineType.NONE,  # MOTOR
        MachineType.NONE,  # SENSOR
        MachineType.CONVEYOR_BELT,  # CONVEYOR_BELT
        MachineType.MINER,  # MINER
        MachineType.ASSEMBLER,  # ASSEMBLER
        MachineType.PALLET,  # PALLET
        MachineType.NONE,  # BASIC_SCIENCE_PACK
        MachineType.NONE,  # ADVANCED_SCIENCE_PACK
        MachineType.ROCKET,  # ROCKET
    ],
    dtype=jnp.int32,
)

MACHINE_TO_ITEM_ARRAY = jnp.array(
    [
        ItemType.EMPTY,  # NONE
        ItemType.MINER,  # MINER
        ItemType.PALLET,  # PALLET
        ItemType.ASSEMBLER,  # ASSEMBLER
        ItemType.CONVEYOR_BELT,  # CONVEYOR_BELT
        ItemType.ROCKET,  # ROCKET
    ],
    dtype=jnp.int32,
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

    # Movement (11)
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

    # Placement — one per placeable machine type (5)
    PLACE_MINER = 11
    PLACE_PALLET = 12
    PLACE_BELT = 13
    PLACE_ASSEMBLER = 14
    PLACE_ROCKET = 15

    # Crafting — one per recipe output (16)
    CRAFT_IRON_PLATE = 16
    CRAFT_COPPER_PLATE = 17
    CRAFT_TIN_PLATE = 18
    CRAFT_WAFER = 19
    CRAFT_STEEL = 20
    CRAFT_CIRCUIT = 21
    CRAFT_WIRE = 22
    CRAFT_MOTOR = 23
    CRAFT_SENSOR = 24
    CRAFT_BELT = 25
    CRAFT_MINER = 26
    CRAFT_ASSEMBLER = 27
    CRAFT_PALLET = 28
    CRAFT_BASIC_SCIENCE = 29
    CRAFT_ADV_SCIENCE = 30
    CRAFT_ROCKET = 31

    # Research (2)
    RESEARCH_BASIC = 32
    RESEARCH_ADVANCED = 33

    # Deposit — one per non-EMPTY item type (21)
    DEPOSIT_COAL = 34
    DEPOSIT_IRON_ORE = 35
    DEPOSIT_COPPER_ORE = 36
    DEPOSIT_TIN_ORE = 37
    DEPOSIT_SILICON = 38
    DEPOSIT_IRON_PLATE = 39
    DEPOSIT_COPPER_PLATE = 40
    DEPOSIT_TIN_PLATE = 41
    DEPOSIT_WAFER = 42
    DEPOSIT_STEEL = 43
    DEPOSIT_CIRCUIT = 44
    DEPOSIT_WIRE = 45
    DEPOSIT_MOTOR = 46
    DEPOSIT_SENSOR = 47
    DEPOSIT_BELT = 48
    DEPOSIT_MINER = 49
    DEPOSIT_ASSEMBLER = 50
    DEPOSIT_PALLET = 51
    DEPOSIT_BASIC_SCIENCE = 52
    DEPOSIT_ADV_SCIENCE = 53
    DEPOSIT_ROCKET = 54

    # Withdraw — one per non-EMPTY item type (21)
    WITHDRAW_COAL = 55
    WITHDRAW_IRON_ORE = 56
    WITHDRAW_COPPER_ORE = 57
    WITHDRAW_TIN_ORE = 58
    WITHDRAW_SILICON = 59
    WITHDRAW_IRON_PLATE = 60
    WITHDRAW_COPPER_PLATE = 61
    WITHDRAW_TIN_PLATE = 62
    WITHDRAW_WAFER = 63
    WITHDRAW_STEEL = 64
    WITHDRAW_CIRCUIT = 65
    WITHDRAW_WIRE = 66
    WITHDRAW_MOTOR = 67
    WITHDRAW_SENSOR = 68
    WITHDRAW_BELT = 69
    WITHDRAW_MINER = 70
    WITHDRAW_ASSEMBLER = 71
    WITHDRAW_PALLET = 72
    WITHDRAW_BASIC_SCIENCE = 73
    WITHDRAW_ADV_SCIENCE = 74
    WITHDRAW_ROCKET = 75

    # Temporary backward-compat actions
    TURN_LEFT = 76
    TURN_RIGHT = 77
    ROTATE = 78
    REPAIR = 79

    # Aliases for old naming convention
    DEPOSIT_ADVANCED_SCIENCE = 53
    WITHDRAW_ADVANCED_SCIENCE = 74


# Base offsets for arithmetic dispatch of compound actions.
PLACE_BASE: int = Action.PLACE_MINER
CRAFT_BASE: int = Action.CRAFT_IRON_PLATE
DEPOSIT_BASE: int = Action.DEPOSIT_COAL
WITHDRAW_BASE: int = Action.WITHDRAW_COAL

# Maps PLACE_* action offset (0..5) to the ItemType of the machine placed.
PLACE_ACTION_TO_ITEM = jnp.array(
    [
        ItemType.MINER,
        ItemType.PALLET,
        ItemType.CONVEYOR_BELT,
        ItemType.ASSEMBLER,
        ItemType.ROCKET,
    ],
    dtype=jnp.int32,
)

# Maps RESEARCH_* action offset (0..1) to the science pack item type.
RESEARCH_ACTION_TO_PACK = jnp.array(
    [
        ItemType.BASIC_SCIENCE_PACK,
        ItemType.ADVANCED_SCIENCE_PACK,
    ],
    dtype=jnp.int32,
)

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
    ],
)

BLOCK_TO_ITEM_ARRAY = jnp.array(
    [
        ItemType.EMPTY,  # INVALID
        ItemType.EMPTY,  # OUT_OF_BOUNDS
        ItemType.EMPTY,  # DIRT
        ItemType.EMPTY,  # WATER
        ItemType.IRON_ORE,  # IRON
        ItemType.COPPER_ORE,  # COPPER
        ItemType.COAL,  # COAL
        ItemType.TIN_ORE,  # TIN
        ItemType.SILICON,  # SILICON
    ],
    dtype=jnp.int32,
)

SOLID_BLOCKS = jnp.array(
    [BlockType.WATER, BlockType.OUT_OF_BOUNDS],
    dtype=jnp.int32,
)

BLOCK_MAX_RESOURCES = 1000

POWER_PER_COAL = 10

MACHINE_POWER_CONSUMPTION = jnp.array(
    [0, 1, 0, 0, 0, 0],
    # NONE, MINER, PALLET, ASM, BELT, ROCKET
    dtype=jnp.int32,
)

MACHINE_MINING_RATE = jnp.array(
    [0, 3, 0, 0, 0, 0],
    # NONE, MINER, PALLET, ASM, BELT, ROCKET
    dtype=jnp.int32,
)

OBS_DIM = (64, 64, 3)
BLOCK_PIXEL_SIZE = 32
NUM_ACTIONS = len(Action)
MAX_ACHIEVEMENTS = 64


# ---------------------------------------------------------------------------
# Temporary backward-compat constants (will be removed in Stage 2+)
# ---------------------------------------------------------------------------

DEFAULT_MACHINE_MAX_HEALTH: int = 100
DEFAULT_MAX_BITERS: int = 32
MAX_ASSEMBLER_STACK_SIZE: int = 1000
NUM_INVENTORY_SLOTS: int = 10
MAX_MACHINE_INVENTORY_SLOTS: int = 8

TURN_LEFT_MAP = jnp.array([0, 4, 3, 1, 2], dtype=jnp.int32)
TURN_RIGHT_MAP = jnp.array([0, 3, 4, 2, 1], dtype=jnp.int32)

MACHINE_TO_RECIPE = jnp.array([-1, 0, 1, 4, 2, -1], dtype=jnp.int32)
TECH_GATES_RECIPE = jnp.array([0, 1], dtype=jnp.int32)

MACHINE_NUM_SLOTS = np.array([0, 2, 1, 4, 1, 0], dtype=np.int32)


class SlotRole(IntEnum):
    """Slot roles for machine inventory display (editor compat)."""

    NONE = 0
    INPUT = 1
    OUTPUT = 2
    STORAGE = 3


MACHINE_SLOT_ROLES = np.array(
    [
        [SlotRole.NONE] * 8,
        [SlotRole.INPUT, SlotRole.OUTPUT] + [SlotRole.NONE] * 6,
        [SlotRole.STORAGE] * 8,
        [SlotRole.INPUT, SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT]
        + [SlotRole.NONE] * 4,
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,
        [SlotRole.NONE] * 8,
    ],
    dtype=np.int32,
)

SLOT_ROLE_LABELS: dict[int, str] = {
    0: "",
    1: "IN",
    2: "OUT",
    3: "STORE",
}
SLOT_ROLE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),
    1: (190, 120, 40),
    2: (40, 170, 140),
    3: (80, 115, 175),
}


def load_texture(name: str) -> np.ndarray:
    """Load a texture from the assets directory.

    Args:
        name: Name of the texture file (without extension).

    Returns:
        RGBA numpy array of shape (BLOCK_PIXEL_SIZE, BLOCK_PIXEL_SIZE, 4).

    Raises:
        FileNotFoundError: If the texture file does not exist.
    """
    import imageio.v3 as iio

    path = ASSETS_PATH / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(f"Texture not found: {path}")
    return iio.imread(path)


def load_all_textures() -> dict[int, np.ndarray]:
    """Load all block textures into a dictionary.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays.
    """
    textures: dict[int, np.ndarray] = {}
    texture_names = {
        BlockType.DIRT: "dirt",
        BlockType.WATER: "water",
        BlockType.IRON: "iron",
        BlockType.COPPER: "copper",
        BlockType.COAL: "coal",
        BlockType.TIN: "tin",
        BlockType.SILICON: "silicon",
    }
    for block_type, name in texture_names.items():
        textures[int(block_type)] = load_texture(name)
    return textures
