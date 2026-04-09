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
    NEST = 7


class ItemType(IntEnum):
    """Item types that can be stored in inventory."""

    EMPTY = 0
    COAL = 1
    IRON = 2
    COPPER = 3
    MINER = 4
    PALLET = 5
    CONVEYOR_BELT = 6
    ARM = 7
    ASSEMBLER = 8
    HULL = 9
    FUEL_PACK = 10
    ROCKET = 11
    BASIC_SCIENCE_PACK = 12
    FUEL_SCIENCE_PACK = 13
    ADVANCED_SCIENCE_PACK = 14


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

# Backward-compat aliases for Phase 2 modules (renderer, editor, play).
# These will be removed when the UI is updated for pouches.
NUM_INVENTORY_SLOTS = 10  # deprecated
MAX_MACHINE_INVENTORY_SLOTS = 8  # deprecated
MACHINE_NUM_SLOTS = np.array(  # deprecated
    [0, 2, 1, 4, 1, 1, 0],
    dtype=np.int32,
)


class SlotRole(IntEnum):  # deprecated — Phase 2 UI still references this
    """Deprecated slot role enum. Kept for Phase 2 UI compat."""

    NONE = 0
    INPUT = 1
    OUTPUT = 2
    STORAGE = 3


MACHINE_SLOT_ROLES: np.ndarray = np.array(  # deprecated
    [
        [SlotRole.NONE] * 8,
        [SlotRole.INPUT, SlotRole.OUTPUT] + [SlotRole.NONE] * 6,
        [SlotRole.STORAGE] * 8,
        [SlotRole.INPUT, SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT]
        + [SlotRole.NONE] * 4,
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,
        [SlotRole.NONE] * 8,
    ],
    dtype=np.int32,
)
SLOT_ROLE_LABELS: dict[int, str] = {  # deprecated
    0: "",
    1: "IN",
    2: "OUT",
    3: "STORE",
}
SLOT_ROLE_COLORS: dict[int, tuple[int, int, int]] = {  # deprecated
    0: (40, 40, 40),
    1: (190, 120, 40),
    2: (40, 170, 140),
    3: (80, 115, 175),
}

# Canonical dtypes for state arrays. Use these in tests and level
# builders to avoid int32/int16 mismatch warnings from JAX scatter ops.
INVENTORY_COUNT_DTYPE = jnp.int32
MACHINE_INVENTORY_COUNT_DTYPE = jnp.int16
BLOCK_RESOURCE_DTYPE = jnp.int16

# Max distinct item types a machine can hold simultaneously.
# Belt/Arm hold one type at a time; Pallet holds 1 type.
MACHINE_MAX_TYPES = jnp.array(
    [0, 2, 1, 4, 1, 1, 0],
    # NONE, MINER, PALLET, ASSEMBLER, BELT, ARM, ROCKET
    dtype=jnp.int32,
)

# Max stack count per item type per machine type.
MACHINE_MAX_STACK = jnp.array(
    [
        0,
        MAX_MACHINE_STACK_SIZE,
        256,
        1000,
        MAX_MACHINE_STACK_SIZE,
        MAX_MACHINE_STACK_SIZE,
        0,
    ],
    # NONE, MINER, PALLET, ASSEMBLER, BELT, ARM, ROCKET
    dtype=jnp.int32,
)

# Max stack count per item type for the player inventory.
PLAYER_MAX_STACK = jnp.array(
    [
        0,  # EMPTY
        64,  # COAL
        64,  # IRON
        64,  # COPPER
        10,  # MINER
        10,  # PALLET
        10,  # CONVEYOR_BELT
        10,  # ARM
        10,  # ASSEMBLER
        64,  # HULL
        64,  # FUEL_PACK
        10,  # ROCKET
        64,  # BASIC_SCIENCE_PACK
        64,  # FUEL_SCIENCE_PACK
        64,  # ADVANCED_SCIENCE_PACK
    ],
    dtype=jnp.int32,
)

BLOCK_TO_ITEM: dict[BlockType, ItemType] = {
    BlockType.COAL: ItemType.COAL,
    BlockType.IRON: ItemType.IRON,
    BlockType.COPPER: ItemType.COPPER,
}

ITEM_COLORS: dict[int, tuple[int, int, int]] = {
    ItemType.COAL: (54, 54, 54),
    ItemType.IRON: (192, 192, 192),
    ItemType.COPPER: (184, 115, 51),
    ItemType.MINER: (0, 200, 0),
    ItemType.PALLET: (140, 100, 45),
    ItemType.CONVEYOR_BELT: (220, 180, 50),
    ItemType.ARM: (80, 120, 200),
    ItemType.ASSEMBLER: (160, 80, 200),
    ItemType.HULL: (170, 170, 190),
    ItemType.FUEL_PACK: (220, 140, 40),
    ItemType.ROCKET: (240, 240, 240),
    ItemType.BASIC_SCIENCE_PACK: (200, 50, 50),
    ItemType.FUEL_SCIENCE_PACK: (50, 150, 50),
    ItemType.ADVANCED_SCIENCE_PACK: (50, 50, 200),
}

# Human-readable display names for each MachineType, used by the UI.
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
# Re-exported here for backward compatibility.
from factoriax.recipes import (  # noqa: E402, F401
    ASSEMBLER_RECIPE_INPUT_COUNTS,
    ASSEMBLER_RECIPE_INPUT_ITEMS,
    ASSEMBLER_RECIPE_NAMES,
    ASSEMBLER_RECIPE_OUTPUTS,
    ASSEMBLER_RECIPE_TICKS,
    ASSEMBLER_RECIPES,
    MAX_ASSEMBLER_RECIPE_INPUTS,
    MAX_ASSEMBLER_STACK_SIZE,
    MAX_RECIPE_INPUTS,
    NUM_ASSEMBLER_RECIPES,
    NUM_RECIPES,
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
# BASIC_SCIENCE_PACK unlocks tech 0, FUEL_SCIENCE_PACK unlocks tech 1.
SCIENCE_PACK_TO_TECH = (
    jnp.zeros(len(ItemType), dtype=jnp.int32)
    .at[ItemType.BASIC_SCIENCE_PACK]
    .set(0)
    .at[ItemType.FUEL_SCIENCE_PACK]
    .set(1)
)

# Whether an item type is a science pack that can be used for research.
IS_RESEARCH_ITEM = (
    jnp.zeros(len(ItemType), dtype=jnp.bool_)
    .at[ItemType.BASIC_SCIENCE_PACK]
    .set(True)
    .at[ItemType.FUEL_SCIENCE_PACK]
    .set(True)
)

# Which assembler recipe index each technology gates.
# Tech 0 gates assembler recipe 0 (Hull), tech 1 gates recipe 1 (Fuel Pack).
TECH_GATES_RECIPE = jnp.array([0, 1], dtype=jnp.int32)

# Which assembler recipes require no research (ungated).
# Recipes not in TECH_GATES_RECIPE are always available.
# Science pack recipes (indices 3, 4, 5) and Rocket (index 2) are ungated.

# ---------------------------------------------------------------------------
# Machine health / repair
# ---------------------------------------------------------------------------

DEFAULT_MACHINE_MAX_HEALTH: int = 100
DEFAULT_MAX_BITERS: int = 32

# Maps MachineType -> player recipe index for repair cost.
# -1 means not repairable (NONE, ROCKET).
MACHINE_TO_RECIPE = jnp.array(
    [
        -1,  # NONE
        0,  # MINER -> recipe 0 (5 copper, 5 iron)
        1,  # PALLET -> recipe 1 (5 iron)
        4,  # ASSEMBLER -> recipe 4 (10 iron, 5 copper)
        2,  # CONVEYOR_BELT -> recipe 2 (1 iron)
        3,  # ARM -> recipe 3 (5 iron, 1 copper)
        -1,  # ROCKET
    ],
    dtype=jnp.int32,
)

PLACEABLE_ITEMS = jnp.array(
    [
        ItemType.MINER,
        ItemType.PALLET,
        ItemType.CONVEYOR_BELT,
        ItemType.ARM,
        ItemType.ASSEMBLER,
        ItemType.ROCKET,
    ],
    dtype=jnp.int32,
)

# Ordered tuple for UI iteration (hotbar tool belt order).
PLACEABLE_ITEM_LIST: tuple[int, ...] = (
    int(ItemType.MINER),
    int(ItemType.PALLET),
    int(ItemType.CONVEYOR_BELT),
    int(ItemType.ARM),
    int(ItemType.ASSEMBLER),
    int(ItemType.ROCKET),
)

PLACEABLE_ITEM_SET: frozenset[int] = frozenset(PLACEABLE_ITEM_LIST)

# Non-placeable item types for the inventory resources section.
RESOURCE_ITEM_LIST: tuple[int, ...] = tuple(
    i for i in range(1, NUM_ITEM_TYPES) if i not in PLACEABLE_ITEM_SET
)

ITEM_TO_MACHINE = {
    ItemType.MINER: MachineType.MINER,
    ItemType.PALLET: MachineType.PALLET,
    ItemType.CONVEYOR_BELT: MachineType.CONVEYOR_BELT,
    ItemType.ARM: MachineType.ARM,
    ItemType.ASSEMBLER: MachineType.ASSEMBLER,
    ItemType.ROCKET: MachineType.ROCKET,
}

ITEM_TO_MACHINE_ARRAY = jnp.array(
    [
        MachineType.NONE,  # EMPTY
        MachineType.NONE,  # COAL
        MachineType.NONE,  # IRON
        MachineType.NONE,  # COPPER
        MachineType.MINER,  # MINER
        MachineType.PALLET,  # PALLET
        MachineType.CONVEYOR_BELT,  # CONVEYOR_BELT
        MachineType.ARM,  # ARM
        MachineType.ASSEMBLER,  # ASSEMBLER
        MachineType.NONE,  # HULL
        MachineType.NONE,  # FUEL_PACK
        MachineType.ROCKET,  # ROCKET
        MachineType.NONE,  # BASIC_SCIENCE_PACK
        MachineType.NONE,  # FUEL_SCIENCE_PACK
        MachineType.NONE,  # ADVANCED_SCIENCE_PACK
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


class Direction(IntEnum):
    """Compass facing directions for players and machines.

    These are stored in ``player_directions`` and ``machine_direction``
    state arrays. They are NOT actions. Use :class:`Action` for agent
    inputs and :class:`Direction` for spatial orientation.
    """

    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class Action(IntEnum):
    """Player actions using compound action design.

    Every action is self-contained: placement, deposit, and withdraw
    actions name the specific item type so no slot cursor is needed.
    Movement actions move in absolute map directions without changing
    facing. FACE_* snaps facing without moving.
    """

    # Movement (11)
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    FACE_UP = 7
    FACE_DOWN = 8
    FACE_LEFT = 9
    FACE_RIGHT = 10

    # World (4)
    MINE = 11
    PICKUP = 12
    ROTATE = 13
    REPAIR = 14

    # Placement — one per placeable machine type (6)
    PLACE_MINER = 15
    PLACE_PALLET = 16
    PLACE_BELT = 17
    PLACE_ARM = 18
    PLACE_ASSEMBLER = 19
    PLACE_ROCKET = 20

    # Crafting — one per player recipe (5)
    CRAFT_MINER = 21
    CRAFT_PALLET = 22
    CRAFT_BELT = 23
    CRAFT_ARM = 24
    CRAFT_ASSEMBLER = 25

    # Research — one per science pack type (3)
    RESEARCH_BASIC = 26
    RESEARCH_FUEL = 27
    RESEARCH_ADVANCED = 28

    # Deposit — one per item type (14)
    DEPOSIT_COAL = 29
    DEPOSIT_IRON = 30
    DEPOSIT_COPPER = 31
    DEPOSIT_MINER = 32
    DEPOSIT_PALLET = 33
    DEPOSIT_BELT = 34
    DEPOSIT_ARM = 35
    DEPOSIT_ASSEMBLER = 36
    DEPOSIT_HULL = 37
    DEPOSIT_FUEL_PACK = 38
    DEPOSIT_ROCKET = 39
    DEPOSIT_BASIC_SCIENCE = 40
    DEPOSIT_FUEL_SCIENCE = 41
    DEPOSIT_ADVANCED_SCIENCE = 42

    # Withdraw — one per item type (14)
    WITHDRAW_COAL = 43
    WITHDRAW_IRON = 44
    WITHDRAW_COPPER = 45
    WITHDRAW_MINER = 46
    WITHDRAW_PALLET = 47
    WITHDRAW_BELT = 48
    WITHDRAW_ARM = 49
    WITHDRAW_ASSEMBLER = 50
    WITHDRAW_HULL = 51
    WITHDRAW_FUEL_PACK = 52
    WITHDRAW_ROCKET = 53
    WITHDRAW_BASIC_SCIENCE = 54
    WITHDRAW_FUEL_SCIENCE = 55
    WITHDRAW_ADVANCED_SCIENCE = 56


# Base offsets for arithmetic dispatch of compound actions.
# item_type = action - DEPOSIT_BASE + ItemType.COAL
PLACE_BASE: int = Action.PLACE_MINER
CRAFT_BASE: int = Action.CRAFT_MINER
DEPOSIT_BASE: int = Action.DEPOSIT_COAL
WITHDRAW_BASE: int = Action.WITHDRAW_COAL

# Maps PLACE_* action offset (0..5) to the ItemType of the machine placed.
PLACE_ACTION_TO_ITEM = jnp.array(
    [
        ItemType.MINER,  # PLACE_MINER - PLACE_BASE = 0
        ItemType.PALLET,  # 1
        ItemType.CONVEYOR_BELT,  # 2
        ItemType.ARM,  # 3
        ItemType.ASSEMBLER,  # 4
        ItemType.ROCKET,  # 5
    ],
    dtype=jnp.int32,
)

# Maps RESEARCH_* action offset (0..2) to the science pack item type.
RESEARCH_ACTION_TO_PACK = jnp.array(
    [
        ItemType.BASIC_SCIENCE_PACK,  # RESEARCH_BASIC
        ItemType.FUEL_SCIENCE_PACK,  # RESEARCH_FUEL
        ItemType.ADVANCED_SCIENCE_PACK,  # RESEARCH_ADVANCED
    ],
    dtype=jnp.int32,
)


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

# Counterclockwise turn: UP→LEFT→DOWN→RIGHT→UP
TURN_LEFT_MAP = jnp.array([0, 4, 3, 1, 2], dtype=jnp.int32)

# Clockwise turn: UP→RIGHT→DOWN→LEFT→UP
TURN_RIGHT_MAP = jnp.array([0, 3, 4, 2, 1], dtype=jnp.int32)

MINEABLE_BLOCKS = jnp.array([BlockType.COAL, BlockType.IRON, BlockType.COPPER])

BLOCK_TO_ITEM_ARRAY = jnp.array(
    [
        ItemType.EMPTY,  # INVALID -> EMPTY
        ItemType.EMPTY,  # OUT_OF_BOUNDS -> EMPTY
        ItemType.EMPTY,  # DIRT -> EMPTY
        ItemType.EMPTY,  # WATER -> EMPTY
        ItemType.IRON,  # IRON -> IRON
        ItemType.COPPER,  # COPPER -> COPPER
        ItemType.COAL,  # COAL -> COAL
    ],
    dtype=jnp.int32,
)

SOLID_BLOCKS = jnp.array(
    [BlockType.WATER, BlockType.OUT_OF_BOUNDS, BlockType.NEST], dtype=jnp.int32
)

BLOCK_MAX_RESOURCES = 1000

POWER_PER_COAL = 10

MACHINE_POWER_CONSUMPTION = jnp.array(
    [0, 1, 0, 0, 0, 0, 0],
    # NONE, MINER, PALLET, ASSEMBLER, CONVEYOR_BELT, ARM, ROCKET
    dtype=jnp.int32,
)

MACHINE_MINING_RATE = jnp.array(
    [0, 3, 0, 0, 0, 0, 0],
    # NONE, MINER, PALLET, ASSEMBLER, CONVEYOR_BELT, ARM, ROCKET
    dtype=jnp.int32,
)

OBS_DIM = (64, 64, 3)
BLOCK_PIXEL_SIZE = 32
NUM_ACTIONS = len(Action)
MAX_ACHIEVEMENTS = 64


def load_texture(name: str) -> np.ndarray:
    """Load a texture from the assets directory.

    Args:
        name: Name of the texture file (without extension)

    Returns:
        RGBA numpy array of shape (BLOCK_PIXEL_SIZE, BLOCK_PIXEL_SIZE, 4)

    Raises:
        FileNotFoundError: If the texture file does not exist
    """
    import imageio.v3 as iio

    path = ASSETS_PATH / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(f"Texture not found: {path}")
    return iio.imread(path)


def load_all_textures() -> dict[int, np.ndarray]:
    """Load all block textures into a dictionary.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    textures: dict[int, np.ndarray] = {}
    texture_names = {
        BlockType.DIRT: "dirt",
        BlockType.WATER: "water",
        BlockType.IRON: "iron",
        BlockType.COPPER: "copper",
        BlockType.COAL: "coal",
    }
    for block_type, name in texture_names.items():
        textures[int(block_type)] = load_texture(name)
    return textures
