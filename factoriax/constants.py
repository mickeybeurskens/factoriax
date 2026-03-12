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


class Action(IntEnum):
    """Player actions."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


DIRECTIONS = jnp.array(
    [
        [0, 0],  # NOOP
        [-1, 0],  # LEFT
        [1, 0],  # RIGHT
        [0, -1],  # UP
        [0, 1],  # DOWN
    ],
    dtype=jnp.int32,
)

SOLID_BLOCKS = jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS], dtype=jnp.int32)

OBS_DIM = (64, 64, 3)
BLOCK_PIXEL_SIZE = 16
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
