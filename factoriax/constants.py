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


NUM_ITEM_TYPES = len(ItemType)
MAX_STACK_SIZE = 64
MAX_MACHINE_STACK_SIZE = 64

# Canonical dtypes for state arrays.
INVENTORY_COUNT_DTYPE = jnp.int32
MACHINE_INVENTORY_COUNT_DTYPE = jnp.int16
BLOCK_RESOURCE_DTYPE = jnp.int16

# Max distinct item types a machine can hold simultaneously.
MACHINE_MAX_TYPES = jnp.array(
    [0, 2, 1, 4, 1, 1, 0, 2, 2],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET, FURNACE, SCIENCE_LAB
    dtype=jnp.int32,
)

# Max stack count per item type per machine type.
MACHINE_MAX_STACK = jnp.array(
    [0, MAX_MACHINE_STACK_SIZE, 256, 1000, 3, 1, 0, 1000, 1000],
    # NONE, MINER, PALLET, ASM, BELT, ARM, ROCKET, FURNACE, SCIENCE_LAB
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
    ItemType.TIN_ORE: (170, 180, 185),
    ItemType.SILICON: (80, 105, 140),
    ItemType.IRON_PLATE: (192, 192, 192),
    ItemType.COPPER_PLATE: (184, 115, 51),
    ItemType.TIN_PLATE: (200, 200, 190),
    ItemType.WAFER: (80, 90, 140),
    ItemType.FRAME: (140, 150, 165),
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
    ItemType.FURNACE: (120, 60, 40),
    ItemType.REFRACTORY: (210, 170, 120),
    ItemType.HULL: (150, 160, 170),
    ItemType.ENGINE_UNIT: (220, 140, 60),
    ItemType.AVIONICS: (80, 200, 220),
    ItemType.ROCKET_CORE: (160, 100, 200),
    # Science lab: deep violet body. Bars/apex/active glow are drawn
    # by the renderer using palette C (see renderer._draw_science_lab_body).
    ItemType.SCIENCE_LAB: (76, 29, 149),
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
    int(MachineType.FURNACE): "Furnace",
    int(MachineType.SCIENCE_LAB): "Science Lab",
}

# Recipes are defined in factoriax.recipes (single source of truth).
from factoriax.recipes import (  # noqa: E402, F401
    CRAFT_ACTION_TO_RECIPE,
    MAX_RECIPE_INPUTS,
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

PLACEABLE_ITEMS = jnp.array(
    [
        ItemType.MINER,
        ItemType.PALLET,
        ItemType.CONVEYOR_BELT,
        ItemType.ASSEMBLER,
        ItemType.ARM,
        ItemType.ROCKET,
        ItemType.FURNACE,
        ItemType.SCIENCE_LAB,
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
    int(ItemType.FURNACE),
    int(ItemType.SCIENCE_LAB),
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
    ItemType.FURNACE: MachineType.FURNACE,
    ItemType.SCIENCE_LAB: MachineType.SCIENCE_LAB,
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
        MachineType.NONE,  # FRAME
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
        MachineType.FURNACE,  # FURNACE
        MachineType.NONE,  # REFRACTORY
        MachineType.NONE,  # HULL
        MachineType.NONE,  # ENGINE_UNIT
        MachineType.NONE,  # AVIONICS
        MachineType.NONE,  # ROCKET_CORE
        MachineType.SCIENCE_LAB,  # SCIENCE_LAB
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
        ItemType.FURNACE,  # FURNACE
        ItemType.SCIENCE_LAB,  # SCIENCE_LAB
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

    # Placement — one per placeable machine type (8)
    PLACE_MINER = 11
    PLACE_PALLET = 12
    PLACE_BELT = 13
    PLACE_ASSEMBLER = 14
    PLACE_ARM = 15
    PLACE_ROCKET = 16
    PLACE_FURNACE = 17
    PLACE_SCIENCE_LAB = 18

    # Crafting — one per recipe output (19)
    CRAFT_IRON_PLATE = 19
    CRAFT_COPPER_PLATE = 20
    CRAFT_TIN_PLATE = 21
    CRAFT_WAFER = 22
    CRAFT_FRAME = 23
    CRAFT_CIRCUIT = 24
    CRAFT_WIRE = 25
    CRAFT_MOTOR = 26
    CRAFT_SENSOR = 27
    CRAFT_BELT = 28
    CRAFT_MINER = 29
    CRAFT_ASSEMBLER = 30
    CRAFT_PALLET = 31
    CRAFT_ARM = 32
    CRAFT_FURNACE = 33
    CRAFT_BASIC_SCIENCE = 34
    CRAFT_ADV_SCIENCE = 35
    CRAFT_ROCKET = 36
    CRAFT_SCIENCE_LAB = 37

    # Deposit — one per non-EMPTY item type (29)
    DEPOSIT_COAL = 38
    DEPOSIT_IRON_ORE = 39
    DEPOSIT_COPPER_ORE = 40
    DEPOSIT_TIN_ORE = 41
    DEPOSIT_SILICON = 42
    DEPOSIT_IRON_PLATE = 43
    DEPOSIT_COPPER_PLATE = 44
    DEPOSIT_TIN_PLATE = 45
    DEPOSIT_WAFER = 46
    DEPOSIT_FRAME = 47
    DEPOSIT_CIRCUIT = 48
    DEPOSIT_WIRE = 49
    DEPOSIT_MOTOR = 50
    DEPOSIT_SENSOR = 51
    DEPOSIT_BELT = 52
    DEPOSIT_MINER = 53
    DEPOSIT_ASSEMBLER = 54
    DEPOSIT_PALLET = 55
    DEPOSIT_ARM = 56
    DEPOSIT_BASIC_SCIENCE = 57
    DEPOSIT_ADV_SCIENCE = 58
    DEPOSIT_ROCKET = 59
    DEPOSIT_FURNACE = 60
    DEPOSIT_REFRACTORY = 61
    DEPOSIT_HULL = 62
    DEPOSIT_ENGINE_UNIT = 63
    DEPOSIT_AVIONICS = 64
    DEPOSIT_ROCKET_CORE = 65
    DEPOSIT_SCIENCE_LAB = 66

    # Withdraw — single action; machines have one output slot so no
    # per-item selection is needed (mirrors PICKUP / MINE).
    WITHDRAW = 67

    # Machine rotation — absolute direction set (4)
    ROTATE_LEFT = 68
    ROTATE_RIGHT = 69
    ROTATE_UP = 70
    ROTATE_DOWN = 71


# Base offsets for arithmetic dispatch of compound actions.
PLACE_BASE: int = Action.PLACE_MINER
CRAFT_BASE: int = Action.CRAFT_IRON_PLATE
DEPOSIT_BASE: int = Action.DEPOSIT_COAL
ROTATE_BASE: int = Action.ROTATE_LEFT

# Maps PLACE_* action offset (0..7) to the ItemType of the machine placed.
PLACE_ACTION_TO_ITEM = jnp.array(
    [
        ItemType.MINER,
        ItemType.PALLET,
        ItemType.CONVEYOR_BELT,
        ItemType.ASSEMBLER,
        ItemType.ARM,
        ItemType.ROCKET,
        ItemType.FURNACE,
        ItemType.SCIENCE_LAB,
    ],
    dtype=jnp.int32,
)

# Maps ROTATE_* action offset (0..3) to Direction values.
ROTATE_ACTION_TO_DIR = jnp.array(
    [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN],
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

OBS_DIM = (64, 64, 3)
BLOCK_PIXEL_SIZE = 32
NUM_ACTIONS = len(Action)
MAX_ACHIEVEMENTS = 64


# ---------------------------------------------------------------------------
# Temporary backward-compat constants (will be removed in Stage 2+)
# ---------------------------------------------------------------------------

DEFAULT_MACHINE_MAX_HEALTH: int = 100
NUM_INVENTORY_SLOTS: int = 10
MAX_MACHINE_INVENTORY_SLOTS: int = 8

TURN_RIGHT_MAP = jnp.array([0, 3, 4, 2, 1], dtype=jnp.int32)

# Per-MachineType slot counts (indexed by MachineType value).
# SCIENCE_LAB: 2 input slots (one per pack type), no output.
MACHINE_NUM_SLOTS = np.array([0, 1, 1, 3, 1, 0, 0, 3, 2], dtype=np.int32)


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
