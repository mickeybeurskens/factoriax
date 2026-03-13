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


class ItemType(IntEnum):
    """Item types that can be stored in inventory."""

    EMPTY = 0
    COAL = 1
    IRON = 2
    COPPER = 3
    MINER = 4


class MachineType(IntEnum):
    """Machine types that can be placed on tiles."""

    NONE = 0
    MINER = 1


NUM_INVENTORY_SLOTS = 10
MAX_STACK_SIZE = 64
MAX_MACHINE_STACK_SIZE = 64
NUM_ITEM_TYPES = len(ItemType)

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
}

RECIPES = [
    {
        "output": ItemType.MINER,
        "inputs": [(ItemType.COPPER, 5), (ItemType.IRON, 5)],
        "ticks": 3,
    },
]

NUM_RECIPES = len(RECIPES)
MAX_RECIPE_INPUTS = 2

RECIPE_NAMES = ["Miner"]

RECIPE_OUTPUTS = jnp.array([ItemType.MINER], dtype=jnp.int32)
RECIPE_TICKS = jnp.array([3], dtype=jnp.int32)
RECIPE_INPUT_ITEMS = jnp.array(
    [[ItemType.COPPER, ItemType.IRON]],
    dtype=jnp.int32,
)
RECIPE_INPUT_COUNTS = jnp.array(
    [[5, 5]],
    dtype=jnp.int32,
)

PLACEABLE_ITEMS = jnp.array([ItemType.MINER], dtype=jnp.int32)

ITEM_TO_MACHINE = {
    ItemType.MINER: MachineType.MINER,
}

ITEM_TO_MACHINE_ARRAY = jnp.array(
    [
        MachineType.NONE,  # EMPTY
        MachineType.NONE,  # COAL
        MachineType.NONE,  # IRON
        MachineType.NONE,  # COPPER
        MachineType.MINER,  # MINER
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
    CRAFT = 6
    PLACE = 7
    NEXT_SLOT = 8
    PREV_SLOT = 9
    NEXT_RECIPE = 10
    PREV_RECIPE = 11


DIRECTIONS = jnp.array(
    [
        [0, 0],  # NOOP
        [-1, 0],  # LEFT
        [1, 0],  # RIGHT
        [0, -1],  # UP
        [0, 1],  # DOWN
        [0, 0],  # MINE (no movement)
        [0, 0],  # CRAFT (no movement)
        [0, 0],  # PLACE (no movement)
        [0, 0],  # NEXT_SLOT (no movement)
        [0, 0],  # PREV_SLOT (no movement)
        [0, 0],  # NEXT_RECIPE (no movement)
        [0, 0],  # PREV_RECIPE (no movement)
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

SOLID_BLOCKS = jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS], dtype=jnp.int32)

BLOCK_MAX_RESOURCES = 100

POWER_PER_COAL = 10

MACHINE_POWER_CONSUMPTION = jnp.array(
    [0, 1],  # NONE=0, MINER=1 power/step
    dtype=jnp.int32,
)

MACHINE_MINING_RATE = jnp.array(
    [0, 3],  # NONE=0, MINER=3 resources/step
    dtype=jnp.int32,
)

OBS_DIM = (64, 64, 3)
BLOCK_PIXEL_SIZE = 32
NUM_ACTIONS = len(Action)


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
