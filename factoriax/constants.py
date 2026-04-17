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
    ARM = 19
    BASIC_SCIENCE_PACK = 20
    ADVANCED_SCIENCE_PACK = 21
    ROCKET = 22


class MachineType(IntEnum):
    """Machine types that can be placed on tiles."""

    NONE = 0
    MINER = 1
    PALLET = 2
    ASSEMBLER = 3
    CONVEYOR_BELT = 4
    ARM = 5
    ROCKET = 6


NUM_ITEM_TYPES = len(ItemType)
MAX_STACK_SIZE = 64
MAX_MACHINE_STACK_SIZE = 64

# Canonical dtypes for state arrays.
INVENTORY_COUNT_DTYPE = jnp.int32
MACHINE_INVENTORY_COUNT_DTYPE = jnp.int16
BLOCK_RESOURCE_DTYPE = jnp.int16

# Max distinct item types a machine can hold simultaneously.
MACHINE_MAX_TYPES = jnp.array(
    [0, 2, 1, 4, 1, 1, 0],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET
    dtype=jnp.int32,
)

# Max stack count per item type per machine type.
MACHINE_MAX_STACK = jnp.array(
    [0, MAX_MACHINE_STACK_SIZE, 256, 1000, 3, 1, 0],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET
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
        1024,  # STEEL
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
    ItemType.ARM: (220, 160, 100),
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
    int(MachineType.ARM): "Arm",
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
        ItemType.ARM,
        ItemType.ROCKET,
    ],
    dtype=jnp.int32,
)

PLACEABLE_ITEM_LIST: tuple[int, ...] = (
    int(ItemType.MINER),
    int(ItemType.PALLET),
    int(ItemType.CONVEYOR_BELT),
    int(ItemType.ASSEMBLER),
    int(ItemType.ARM),
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
    ItemType.ARM: MachineType.ARM,
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
        MachineType.ARM,  # ARM
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
        ItemType.ARM,  # ARM
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

    # Placement — one per placeable machine type (6)
    PLACE_MINER = 11
    PLACE_PALLET = 12
    PLACE_BELT = 13
    PLACE_ASSEMBLER = 14
    PLACE_ARM = 15
    PLACE_ROCKET = 16

    # Crafting — one per recipe output (17, includes ARM)
    CRAFT_IRON_PLATE = 17
    CRAFT_COPPER_PLATE = 18
    CRAFT_TIN_PLATE = 19
    CRAFT_WAFER = 20
    CRAFT_STEEL = 21
    CRAFT_CIRCUIT = 22
    CRAFT_WIRE = 23
    CRAFT_MOTOR = 24
    CRAFT_SENSOR = 25
    CRAFT_BELT = 26
    CRAFT_MINER = 27
    CRAFT_ASSEMBLER = 28
    CRAFT_PALLET = 29
    CRAFT_ARM = 30
    CRAFT_BASIC_SCIENCE = 31
    CRAFT_ADV_SCIENCE = 32
    CRAFT_ROCKET = 33

    # Research (2)
    RESEARCH_BASIC = 34
    RESEARCH_ADVANCED = 35

    # Deposit — one per non-EMPTY item type (22)
    DEPOSIT_COAL = 36
    DEPOSIT_IRON_ORE = 37
    DEPOSIT_COPPER_ORE = 38
    DEPOSIT_TIN_ORE = 39
    DEPOSIT_SILICON = 40
    DEPOSIT_IRON_PLATE = 41
    DEPOSIT_COPPER_PLATE = 42
    DEPOSIT_TIN_PLATE = 43
    DEPOSIT_WAFER = 44
    DEPOSIT_STEEL = 45
    DEPOSIT_CIRCUIT = 46
    DEPOSIT_WIRE = 47
    DEPOSIT_MOTOR = 48
    DEPOSIT_SENSOR = 49
    DEPOSIT_BELT = 50
    DEPOSIT_MINER = 51
    DEPOSIT_ASSEMBLER = 52
    DEPOSIT_PALLET = 53
    DEPOSIT_ARM = 54
    DEPOSIT_BASIC_SCIENCE = 55
    DEPOSIT_ADV_SCIENCE = 56
    DEPOSIT_ROCKET = 57

    # Withdraw — one per non-EMPTY item type (22)
    WITHDRAW_COAL = 58
    WITHDRAW_IRON_ORE = 59
    WITHDRAW_COPPER_ORE = 60
    WITHDRAW_TIN_ORE = 61
    WITHDRAW_SILICON = 62
    WITHDRAW_IRON_PLATE = 63
    WITHDRAW_COPPER_PLATE = 64
    WITHDRAW_TIN_PLATE = 65
    WITHDRAW_WAFER = 66
    WITHDRAW_STEEL = 67
    WITHDRAW_CIRCUIT = 68
    WITHDRAW_WIRE = 69
    WITHDRAW_MOTOR = 70
    WITHDRAW_SENSOR = 71
    WITHDRAW_BELT = 72
    WITHDRAW_MINER = 73
    WITHDRAW_ASSEMBLER = 74
    WITHDRAW_PALLET = 75
    WITHDRAW_ARM = 76
    WITHDRAW_BASIC_SCIENCE = 77
    WITHDRAW_ADV_SCIENCE = 78
    WITHDRAW_ROCKET = 79

    # Machine rotation — absolute direction set (4)
    ROTATE_LEFT = 80
    ROTATE_RIGHT = 81
    ROTATE_UP = 82
    ROTATE_DOWN = 83


# Base offsets for arithmetic dispatch of compound actions.
PLACE_BASE: int = Action.PLACE_MINER
CRAFT_BASE: int = Action.CRAFT_IRON_PLATE
DEPOSIT_BASE: int = Action.DEPOSIT_COAL
WITHDRAW_BASE: int = Action.WITHDRAW_COAL
ROTATE_BASE: int = Action.ROTATE_LEFT

# Maps PLACE_* action offset (0..5) to the ItemType of the machine placed.
PLACE_ACTION_TO_ITEM = jnp.array(
    [
        ItemType.MINER,
        ItemType.PALLET,
        ItemType.CONVEYOR_BELT,
        ItemType.ASSEMBLER,
        ItemType.ARM,
        ItemType.ROCKET,
    ],
    dtype=jnp.int32,
)

# Maps ROTATE_* action offset (0..3) to Direction values.
ROTATE_ACTION_TO_DIR = jnp.array(
    [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN],
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

MACHINE_POWER_CONSUMPTION = jnp.array(
    [0, 1, 0, 0, 0, 0, 0],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET
    dtype=jnp.int32,
)

MACHINE_MINING_RATE = jnp.array(
    [0, 3, 0, 0, 0, 0, 0],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET
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

MACHINE_TO_RECIPE = jnp.array([-1, 0, 1, 4, 2, -1, -1], dtype=jnp.int32)
TECH_GATES_RECIPE = jnp.array([0, 1], dtype=jnp.int32)

MACHINE_NUM_SLOTS = np.array([0, 2, 1, 3, 1, 0, 0], dtype=np.int32)


class SlotRole(IntEnum):
    """Slot roles for machine inventory display (editor compat)."""

    NONE = 0
    INPUT = 1
    OUTPUT = 2
    STORAGE = 3
    FUEL = 4


MACHINE_SLOT_ROLES = np.array(
    [
        [SlotRole.NONE] * 8,
        [SlotRole.FUEL, SlotRole.OUTPUT] + [SlotRole.NONE] * 6,
        [SlotRole.STORAGE] * 8,
        [SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT] + [SlotRole.NONE] * 5,
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,
        [SlotRole.NONE] * 8,  # ARM (instant, no buffer)
        [SlotRole.NONE] * 8,  # ROCKET
    ],
    dtype=np.int32,
)

SLOT_ROLE_LABELS: dict[int, str] = {
    0: "",
    1: "IN",
    2: "OUT",
    3: "STORE",
    4: "FUEL",
}
SLOT_ROLE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),
    1: (190, 120, 40),
    2: (40, 170, 140),
    3: (80, 115, 175),
    4: (200, 60, 60),
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
