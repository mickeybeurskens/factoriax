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
    CHEST = 5
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
    CHEST = 2
    ASSEMBLER = 3
    CONVEYOR_BELT = 4
    ARM = 5
    ROCKET = 6


class SlotRole(IntEnum):
    """Role of a machine inventory slot, describing automated item flow direction.

    Roles determine which agents (machine logic vs player) may read or write
    a slot.  The player may withdraw from *any* role; the player may only
    *deposit* into INPUT or STORAGE — never into OUTPUT.  Machine logic may
    only write to OUTPUT and read from INPUT.
    """

    NONE = 0  # Unused / padding slot — no automated flow, no player access.
    INPUT = 1  # Machine consumes from here; player may deposit and withdraw.
    OUTPUT = 2  # Machine produces here; player may only withdraw.
    STORAGE = 3  # No automated flow; player may deposit and withdraw freely.


NUM_INVENTORY_SLOTS = 10
MAX_STACK_SIZE = 64
MAX_MACHINE_STACK_SIZE = 64
MAX_MACHINE_INVENTORY_SLOTS = 8
NUM_ITEM_TYPES = len(ItemType)

# Per-slot roles for each MachineType,
# shape (NUM_MACHINE_TYPES, MAX_MACHINE_INVENTORY_SLOTS).
# Indexed as MACHINE_SLOT_ROLES[machine_type, slot_index].
MACHINE_SLOT_ROLES: np.ndarray = np.array(
    [
        # NONE — no slots active
        [SlotRole.NONE] * 8,
        # MINER — slot 0: fuel input, slot 1: ore output
        [SlotRole.INPUT, SlotRole.OUTPUT] + [SlotRole.NONE] * 6,
        # CHEST — all 8 slots are general storage
        [SlotRole.STORAGE] * 8,
        # ASSEMBLER — slots 0-2: ingredient inputs, slot 3: product output
        [SlotRole.INPUT, SlotRole.INPUT, SlotRole.INPUT, SlotRole.OUTPUT]
        + [SlotRole.NONE] * 4,
        # CONVEYOR_BELT — slot 0: single storage buffer
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,
        # ARM — slot 0: single storage buffer (pick/deposit working buffer)
        [SlotRole.STORAGE] + [SlotRole.NONE] * 7,
        # ROCKET — no slots
        [SlotRole.NONE] * 8,
    ],
    dtype=np.int32,
)

# Number of active (non-NONE) slots per machine type.
MACHINE_NUM_SLOTS: np.ndarray = np.array([0, 2, 8, 4, 1, 1, 0], dtype=np.int32)

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
    ItemType.CHEST: (210, 190, 50),
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
    int(MachineType.CHEST): "Chest",
    int(MachineType.ASSEMBLER): "Assembler",
    int(MachineType.CONVEYOR_BELT): "Conveyor Belt",
    int(MachineType.ARM): "Arm",
    int(MachineType.ROCKET): "Rocket",
}

# Short badge labels for each SlotRole, rendered inside the slot cell header.
SLOT_ROLE_LABELS: dict[int, str] = {
    int(SlotRole.NONE): "",
    int(SlotRole.INPUT): "IN",
    int(SlotRole.OUTPUT): "OUT",
    int(SlotRole.STORAGE): "STORE",
}

# Badge background colours per SlotRole: amber=INPUT, teal=OUTPUT, steel=STORAGE.
SLOT_ROLE_COLORS: dict[int, tuple[int, int, int]] = {
    int(SlotRole.NONE): (40, 40, 40),
    int(SlotRole.INPUT): (190, 120, 40),
    int(SlotRole.OUTPUT): (40, 170, 140),
    int(SlotRole.STORAGE): (80, 115, 175),
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
# BASIC_SCIENCE_PACK unlocks tech 0 (Hull), FUEL_SCIENCE_PACK unlocks tech 1 (Fuel Pack).
SCIENCE_PACK_TO_TECH = jnp.zeros(len(ItemType), dtype=jnp.int32).at[
    ItemType.BASIC_SCIENCE_PACK
].set(0).at[
    ItemType.FUEL_SCIENCE_PACK
].set(1)

# Whether an item type is a science pack that can be used for research.
IS_RESEARCH_ITEM = jnp.zeros(len(ItemType), dtype=jnp.bool_).at[
    ItemType.BASIC_SCIENCE_PACK
].set(True).at[
    ItemType.FUEL_SCIENCE_PACK
].set(True)

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
        0,   # MINER -> recipe 0 (5 copper, 5 iron)
        1,   # CHEST -> recipe 1 (5 iron)
        4,   # ASSEMBLER -> recipe 4 (10 iron, 5 copper)
        2,   # CONVEYOR_BELT -> recipe 2 (1 iron)
        3,   # ARM -> recipe 3 (5 iron, 1 copper)
        -1,  # ROCKET
    ],
    dtype=jnp.int32,
)

PLACEABLE_ITEMS = jnp.array(
    [
        ItemType.MINER,
        ItemType.CHEST,
        ItemType.CONVEYOR_BELT,
        ItemType.ARM,
        ItemType.ASSEMBLER,
        ItemType.ROCKET,
    ],
    dtype=jnp.int32,
)

ITEM_TO_MACHINE = {
    ItemType.MINER: MachineType.MINER,
    ItemType.CHEST: MachineType.CHEST,
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
        MachineType.CHEST,  # CHEST
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
        ItemType.CHEST,  # CHEST
        ItemType.ASSEMBLER,  # ASSEMBLER
        ItemType.CONVEYOR_BELT,  # CONVEYOR_BELT
        ItemType.ARM,  # ARM
        ItemType.ROCKET,  # ROCKET
    ],
    dtype=jnp.int32,
)


class Action(IntEnum):
    """Player actions."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    MINE = 5
    PLACE = 6
    NEXT_SLOT = 7
    PREV_SLOT = 8
    PICKUP = 9
    DEPOSIT = 10
    WITHDRAW = 11
    ROTATE = 12
    NEXT_MACHINE_SLOT = 13
    PREV_MACHINE_SLOT = 14
    CRAFT_MINER = 15
    CRAFT_CHEST = 16
    CRAFT_BELT = 17
    CRAFT_ARM = 18
    CRAFT_ASSEMBLER = 19
    RESEARCH = 20
    REPAIR = 21


DIRECTIONS = jnp.array(
    [
        [0, 0],  # NOOP
        [-1, 0],  # LEFT
        [1, 0],  # RIGHT
        [0, -1],  # UP
        [0, 1],  # DOWN
        [0, 0],  # MINE
        [0, 0],  # PLACE
        [0, 0],  # NEXT_SLOT
        [0, 0],  # PREV_SLOT
        [0, 0],  # PICKUP
        [0, 0],  # DEPOSIT
        [0, 0],  # WITHDRAW
        [0, 0],  # ROTATE
        [0, 0],  # NEXT_MACHINE_SLOT
        [0, 0],  # PREV_MACHINE_SLOT
        [0, 0],  # CRAFT_MINER
        [0, 0],  # CRAFT_CHEST
        [0, 0],  # CRAFT_BELT
        [0, 0],  # CRAFT_ARM
        [0, 0],  # CRAFT_ASSEMBLER
        [0, 0],  # RESEARCH
        [0, 0],  # REPAIR
    ],
    dtype=jnp.int32,
)

DIRECTION_OFFSETS = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

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
    # NONE, MINER, CHEST, ASSEMBLER, CONVEYOR_BELT, ARM, ROCKET
    dtype=jnp.int32,
)

MACHINE_MINING_RATE = jnp.array(
    [0, 3, 0, 0, 0, 0, 0],
    # NONE, MINER, CHEST, ASSEMBLER, CONVEYOR_BELT, ARM, ROCKET
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
