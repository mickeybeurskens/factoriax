"""Pixel rendering for the FactoriaX environment."""

import numpy as np

from factoriax.constants import BLOCK_PIXEL_SIZE, BlockType, load_all_textures
from factoriax.state import EnvState


def create_default_textures() -> dict[int, np.ndarray]:
    """Create simple default textures if asset files don't exist.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    size = BLOCK_PIXEL_SIZE
    textures: dict[int, np.ndarray] = {}

    dirt = np.zeros((size, size, 4), dtype=np.uint8)
    dirt[:, :, 0] = 139
    dirt[:, :, 1] = 90
    dirt[:, :, 2] = 43
    dirt[:, :, 3] = 255
    textures[int(BlockType.DIRT)] = dirt

    water = np.zeros((size, size, 4), dtype=np.uint8)
    water[:, :, 0] = 64
    water[:, :, 1] = 164
    water[:, :, 2] = 223
    water[:, :, 3] = 255
    textures[int(BlockType.WATER)] = water

    return textures


def create_player_texture() -> np.ndarray:
    """Create a simple player texture.

    Returns:
        RGBA numpy array of shape (BLOCK_PIXEL_SIZE, BLOCK_PIXEL_SIZE, 4)
    """
    size = BLOCK_PIXEL_SIZE
    player = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    radius = size // 3
    for y in range(size):
        for x in range(size):
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            if dist <= radius:
                player[y, x] = [255, 100, 100, 255]
    return player


def get_textures() -> dict[int, np.ndarray]:
    """Load textures from files, falling back to defaults if not found.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    try:
        return load_all_textures()
    except FileNotFoundError:
        return create_default_textures()


def render_pixels(
    state: EnvState, block_pixel_size: int = BLOCK_PIXEL_SIZE
) -> np.ndarray:
    """Render the environment state as an RGB pixel image.

    Args:
        state: Current environment state
        block_pixel_size: Size of each block in pixels

    Returns:
        RGB numpy array of the rendered scene
    """
    textures = get_textures()
    player_texture = create_player_texture()

    map_array = np.array(state.map)
    map_height, map_width = map_array.shape
    img_height = map_height * block_pixel_size
    img_width = map_width * block_pixel_size
    image = np.zeros((img_height, img_width, 4), dtype=np.uint8)

    for y in range(map_height):
        for x in range(map_width):
            block_type = int(map_array[y, x])
            if block_type in textures:
                texture = textures[block_type]
            else:
                texture = textures[int(BlockType.DIRT)]

            y_start = y * block_pixel_size
            x_start = x * block_pixel_size
            image[
                y_start : y_start + block_pixel_size,
                x_start : x_start + block_pixel_size,
            ] = texture

    player_pos = np.array(state.player_position)
    px, py = int(player_pos[0]), int(player_pos[1])
    py_start = py * block_pixel_size
    px_start = px * block_pixel_size
    _alpha_blend_inplace(
        image,
        player_texture,
        py_start,
        px_start,
        block_pixel_size,
    )

    return image[:, :, :3]


def _alpha_blend_inplace(
    background: np.ndarray,
    foreground: np.ndarray,
    y_start: int,
    x_start: int,
    size: int,
) -> None:
    """Blend a foreground texture onto the background using alpha compositing.

    Args:
        background: RGBA background image to modify in place
        foreground: RGBA foreground texture
        y_start: Y coordinate of top-left corner
        x_start: X coordinate of top-left corner
        size: Size of the foreground texture
    """
    fg_alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    bg_region = background[y_start : y_start + size, x_start : x_start + size]
    blended = foreground[:, :, :3].astype(np.float32) * fg_alpha + bg_region[
        :, :, :3
    ].astype(np.float32) * (1 - fg_alpha)
    bg_region[:, :, :3] = blended.astype(np.uint8)
    bg_region[:, :, 3] = 255
